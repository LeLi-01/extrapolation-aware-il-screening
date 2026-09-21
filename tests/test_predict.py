"""
基础自动化测试：
1.合法SMILES能通过RDKit验证
2.非法SMILES会被拒绝
3.新样本经过训练时的mask+scaler后，特征维度一致
4.四个模型都能完成一次预测，且输出为有限数值
5.相同输入重复预测结果一致

运行方式（在项目根目录PythonProject2/下）：
    pytest -q
或者：
    pytest tests/test_predict.py -q
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pytest


from pathlib import Path
import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

from src.preprocess import (
    cation_smiles,
    anion_smiles,
    validate_smiles,
    compute_descriptors,
)


#1.测试使用的固定样本
TEST_CATION = "C4MIm or BMIM"
TEST_ANION = "OAc or CH3COO"
TEST_T = 303.15
TEST_P = 1.0

EXPECTED_MODELS = {"XGBoost", "RF", "SVR", "GPR"}

#2.公共fixture：整组测试只加载/计算一次
@pytest.fixture(scope="module")
def inference_assets():
    """
    加载预测阶段真正需要的持久化资产。
    如果文件不存在，测试直接给出明确错误。
    """
    scaler_path = MODELS_DIR / "scaler.pkl"
    nan_mask_path = MODELS_DIR / "nan_cols_mask.npy"
    var_mask_path = MODELS_DIR / "var_mask.npy"
    models_path = MODELS_DIR / "vs_models.pkl"

    required_files = [
        scaler_path,
        nan_mask_path,
        var_mask_path,
        models_path,
    ]

    missing = [str(p) for p in required_files if not p.exists()]
    assert not missing, (
        "以下预测资产不存在，请先运行 preprocess.py / train.py 生成：\n"
        + "\n".join(missing)
    )

    scaler = joblib.load(scaler_path)
    nan_cols = np.load(nan_mask_path)
    var_mask = np.load(var_mask_path)
    models = joblib.load(models_path)

    return {
        "scaler": scaler,
        "nan_cols": nan_cols,
        "var_mask": var_mask,
        "models": models,
    }

@pytest.fixture(scope="module")
def descriptor_tables():
    """
    RDKit 描述符只计算一次，避免每个测试重复计算。
    """
    cation_df = compute_descriptors(cation_smiles)
    anion_df = compute_descriptors(anion_smiles)

    return cation_df, anion_df

def build_test_feature(inference_assets, descriptor_tables):
    """
    按 predict.py 当前的真实顺序构造一个新 IL 样本：
    阳离子描述符 + 阴离子描述符 + T + P
        -> nan_cols mask
        -> var_mask
        -> scaler.transform
    """
    cation_df, anion_df = descriptor_tables

    cation_vec = cation_df.loc[TEST_CATION].to_numpy()
    anion_vec = anion_df.loc[TEST_ANION].to_numpy()

    feat = np.concatenate(
        [cation_vec, anion_vec, [TEST_T, TEST_P]]
    )

    feat = np.nan_to_num(
        feat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    nan_cols = inference_assets["nan_cols"]
    var_mask = inference_assets["var_mask"]
    scaler = inference_assets["scaler"]

    #复用训练阶段保存下来的两个mask
    feat = feat[~nan_cols]
    feat = feat[var_mask]

    feat_scaled = scaler.transform(
        feat.reshape(1, -1)
    )

    return feat_scaled

#3.Test1：正常输入
def test_valid_smiles():
    smiles = cation_smiles[TEST_CATION]
    assert validate_smiles(smiles) is True

#4.Test2：异常输入
def test_invalid_smiles():
    invalid_smiles = "this_is_not_a_valid_smiles"
    assert validate_smiles(invalid_smiles) is False

#5.Test3：训练/预测特征维度必须一致
def test_feature_dimension(inference_assets, descriptor_tables):
    feat_scaled = build_test_feature(
        inference_assets,
        descriptor_tables,
    )

    scaler = inference_assets["scaler"]

    assert feat_scaled.ndim == 2
    assert feat_scaled.shape[0] == 1
    assert feat_scaled.shape[1] == scaler.n_features_in_


#6.Test4：四个模型均可正常加载并预测
def test_all_models_can_predict(inference_assets, descriptor_tables):
    feat_scaled = build_test_feature(
        inference_assets,
        descriptor_tables,
    )
    models = inference_assets["models"]
    # 先检查模型字典的key，避免再次出现KeyError:'XGBoost'
    assert EXPECTED_MODELS.issubset(models.keys()), (
        f"vs_models.pkl 中的模型为 {list(models.keys())}，"
        f"但测试期望至少包含 {sorted(EXPECTED_MODELS)}"
    )

    for model_name in EXPECTED_MODELS:
        pred = models[model_name].predict(feat_scaled)

        assert len(pred) == 1
        assert np.isfinite(pred[0]), (
            f"{model_name} 返回了非有限预测值: {pred[0]}"
        )

#7.Test5：同一输入重复预测应一致

def test_prediction_is_reproducible(inference_assets, descriptor_tables):
    feat_scaled = build_test_feature(
        inference_assets,
        descriptor_tables,
    )

    models = inference_assets["models"]

    for model_name in EXPECTED_MODELS:
        p1 = models[model_name].predict(feat_scaled)[0]
        p2 = models[model_name].predict(feat_scaled)[0]

        assert np.isclose(p1, p2), (
            f"{model_name} 对相同输入的两次预测不一致: "
            f"{p1} vs {p2}"
        )
