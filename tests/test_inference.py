"""
tests/test_inference.py

验证 src/inference.py 统一推理接口：
1. 特征维度/顺序与训练阶段一致（436 -> 201）。
2. 描述符顺序锚点与运行时 CalcMolDescriptors 一致。
3. 与旧预测管线（根 predict.py / src/predict.py 的 models/vs_models.pkl 路径）
   对相同样本给出数值一致的预测（金标准回归）。
4. 可靠分级三档：内插（组合已知）/ Tier 1（组分已知、组合新）/ 训练集外。
5. 非法 SMILES 被拒绝；相同输入可重复。

运行方式（项目根目录 PythonProject2/ 下）：
    pytest tests/test_inference.py -q
"""
import numpy as np
import pytest

from src.inference import (
    DESCRIPTOR_NAMES_PATH,
    MODEL_NAMES,
    _DESCRIPTOR_NAMES,
    build_feature,
    known_status,
    predict_il,
    predict_batch,
    validate_input,
    check_physical_range,
    risk_status,
    _get_assets,
)

# 金标准基线：由旧预测管线于 2026-10-27 采集（两条旧路径结果完全一致）。
# [cation][anion] T P -> {XGBoost, RF, SVR, GPR, mean}
GOLDEN = [
    (
        "C4MIm or BMIM", "TFSI or NTf2", 303.15, 1.0,
        {"XGBoost": 0.07795625, "RF": 0.06468535, "SVR": 0.15100035,
         "GPR": 0.16040925, "mean": 0.11351280},
    ),
    (
        "C4MIm or BMIM", "OAc or CH3COO", 303.15, 1.0,
        {"XGBoost": 0.38654265, "RF": 0.39976741, "SVR": 0.39749475,
         "GPR": 0.36124442, "mean": 0.38626231},
    ),
    (
        "EMIM or C2MIm", "BF4", 313.15, 2.0,
        {"XGBoost": 0.09394292, "RF": 0.11506962, "SVR": 0.09491070,
         "GPR": 0.10171189, "mean": 0.10140878},
    ),
]


def test_feature_alignment():
    """训练阶段维度：原始 436 -> 去 NaN 424 -> 去零方差 201 == scaler.n_features_in_。"""
    assets = _get_assets()
    assert assets["nan_cols"].size == 436
    assert int((~assets["nan_cols"]).sum()) == 424
    assert int(assets["var_mask"].sum()) == 201
    assert assets["scaler"].n_features_in_ == 201

    feat = build_feature("C4MIm or BMIM", "TFSI or NTf2", 303.15, 1.0)
    assert feat.ndim == 2
    assert feat.shape == (1, 201)


def test_descriptor_anchor_consistency():
    """描述符顺序锚点存在，且与运行时 CalcMolDescriptors 逐元素一致。"""
    assert len(_DESCRIPTOR_NAMES) == 217
    if DESCRIPTOR_NAMES_PATH.exists():
        saved = np.load(DESCRIPTOR_NAMES_PATH, allow_pickle=True).tolist()
        assert saved == _DESCRIPTOR_NAMES


@pytest.mark.parametrize("cat,ani,T,P,expected", GOLDEN)
def test_predict_matches_golden(cat, ani, T, P, expected):
    result = predict_il(cat, ani, T, P)
    for key in MODEL_NAMES:
        assert np.isclose(result[key], expected[key], rtol=1e-6, atol=1e-6), (
            f"{key} 与旧管线不一致: {result[key]} vs {expected[key]}"
        )
    assert np.isclose(result["mean"], expected["mean"], rtol=1e-6, atol=1e-6)


def test_short_name_equals_full_name():
    """短名（如 BMIM/TFSI）应与全名解析到同一结果。"""
    full = predict_il("C4MIm or BMIM", "TFSI or NTf2", 303.15, 1.0)
    short = predict_il("BMIM", "TFSI", 303.15, 1.0)
    for key in MODEL_NAMES:
        assert np.isclose(full[key], short[key], rtol=1e-12, atol=1e-12)


def test_raw_smiles_equals_known_name():
    """输入已知离子的原始 SMILES 应与按名称输入结果一致（精确匹配）。"""
    cation_smi = "CCCC[n+]1ccn(C)c1"        # [C4MIm]
    anion_smi = "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"  # [TFSI]
    by_name = predict_il("C4MIm or BMIM", "TFSI or NTf2", 303.15, 1.0)
    by_smiles = predict_il(cation_smi, anion_smi, 303.15, 1.0)
    for key in MODEL_NAMES:
        assert np.isclose(by_name[key], by_smiles[key], rtol=1e-12, atol=1e-12)
    # 精确 SMILES 命中应视为已知离子，且组合已知（内插）
    assert by_smiles["cation_known"] is True
    assert by_smiles["anion_known"] is True
    assert by_smiles["pair_measured"] is True


def test_measured_pair_is_interpolation():
    """完整组合在训练集出现 -> 内插。"""
    status = known_status("C4MIm or BMIM", "TFSI or NTf2")
    assert status["pair_measured"] is True
    assert status["tier"] == "内插（组合已知）"


def test_tier1_is_known_components_new_pair():
    """组分已知但组合未测量 -> Tier 1（外推），如 [EMIM][BF4]。"""
    status = known_status("EMIM or C2MIm", "BF4")
    assert status["cation_known"] is True
    assert status["anion_known"] is True
    assert status["pair_measured"] is False
    assert status["tier"] == "Tier 1"


def test_unknown_ion_flagged():
    """未知离子（训练集外）应明确标记，不得套用 Tier 1。"""
    result = predict_il("CCCC[n+]1ccn(C)c1", "[S-]C#N", 303.15, 1.0)  # SCN 为新阴离子
    assert result["anion_known"] is False
    assert result["pair_measured"] is False
    assert result["tier"] == "训练集外"
    assert "无法按现有 Tier 规则可靠分级" in result["tier_note"]


def test_invalid_smiles_rejected():
    with pytest.raises(ValueError):
        predict_il("not_a_valid_smiles", "BF4", 303.15, 1.0)


def test_prediction_reproducible():
    a = predict_il("C4MIm or BMIM", "TFSI or NTf2", 303.15, 1.0)
    b = predict_il("C4MIm or BMIM", "TFSI or NTf2", 303.15, 1.0)
    for key in MODEL_NAMES:
        assert np.isclose(a[key], b[key])


# ============================================================
# Phase 3：批量输入校验 / 风险判定 / 批量预测
# ============================================================
def test_validate_invalid_smiles():
    errs = validate_input("not_a_valid_smiles", "BF4", 303.15, 1.0)
    assert errs
    assert any("SMILES" in e for e in errs)


def test_validate_missing_input():
    # 空字符串阴离子
    assert validate_input("BMIM", "", 303.15, 1.0)
    # None 阳离子
    assert validate_input(None, "TFSI", 303.15, 1.0)
    # NaN 温度
    assert validate_input("BMIM", "TFSI", float("nan"), 1.0)


def test_validate_negative_pressure():
    errs = validate_input("BMIM", "TFSI", 303.15, -1.0)
    assert any("压力" in e and "大于 0" in e for e in errs)


def test_validate_negative_temperature():
    errs = validate_input("BMIM", "TFSI", -10.0, 1.0)
    assert any("温度" in e for e in errs)


def test_validate_unknown_but_valid_smiles_is_accepted():
    # 未知但可解析的合法 SMILES 不应被当作非法输入（由 known_status 标记为训练集外）
    assert validate_input("CCCC[n+]1ccn(C)c1", "[S-]C#N", 303.15, 1.0) == []


def test_check_physical_range_anomaly():
    # T=423.15 时 SVR/GPR 预测为负
    r = predict_il("C4MIm or BMIM", "TFSI or NTf2", 423.15, 1.0)
    assert set(check_physical_range(r)) == {"SVR", "GPR"}
    assert risk_status(r) == "物理范围异常"


def test_risk_status_extrapolation():
    # T=280 低于训练下限，但四模型预测仍落在 [0,1]
    r = predict_il("C4MIm or BMIM", "TFSI or NTf2", 280.0, 1.0)
    assert r["T_in_training_range"] is False
    assert check_physical_range(r) == []
    assert risk_status(r) == "工况外推"


def test_risk_status_normal():
    r = predict_il("C4MIm or BMIM", "TFSI or NTf2", 303.15, 1.0)
    assert check_physical_range(r) == []
    assert risk_status(r) == "正常"


def test_predict_batch_mixed_results_and_errors():
    records = [
        {"Cation": "C4MIm or BMIM", "Anion": "TFSI or NTf2", "T": 303.15, "P": 1.0},
        {"Cation": "not_a_valid", "Anion": "BF4", "T": 303.15, "P": 1.0},
        {"Cation": "BMIM", "Anion": "TFSI", "T": 303.15, "P": -1.0},
    ]
    results, errors = predict_batch(records)
    assert len(results) == 1
    assert len(errors) == 2

    r0 = results[0]
    assert r0["risk_status"] == "正常"
    assert r0["out_of_range_models"] == []
    assert r0["tier"] == "内插（组合已知）"

    assert errors[0]["row"] == 2
    assert any("SMILES" in e for e in errors[0]["errors"])
    assert errors[1]["row"] == 3
    assert any("压力" in e for e in errors[1]["errors"])
