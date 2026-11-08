"""
统一推理接口。

使用 models/ 目录下已持久化的四模型 (vs_models.pkl)、StandardScaler (scaler.pkl)
与两个特征掩码 (nan_cols_mask.npy / var_mask.npy) 进行推理，无需重新训练。

单位约定：
    T     —— 开尔文 (K)，训练范围 293.15–403.15
    P     —— bar，训练范围 0.001–96.3
    输出  —— H2S 溶解度（摩尔分数，无量纲）

注意：本模型预测的是 H2S 溶解度，仅作为筛选硫醇前置吸收离子液体 (IL) 的代理指标，
不代表硫醇溶解度本身。

设计要点：
    - 惰性加载模型资产，import 本模块无重副作用（区别于根目录 predict.py 的模块级加载）。
    - 所有路径基于 __file__ 定位，不依赖当前工作目录。
    - 描述符顺序以 models/descriptor_names.npy 为锚点，并与运行时 CalcMolDescriptors
      逐元素校验，避免 RDKit 版本漂移导致的特征错位。
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.preprocess import anion_smiles, cation_smiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

CLEAN_CSV = DATA_DIR / "zhao_data.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
NAN_COLS_PATH = MODELS_DIR / "nan_cols_mask.npy"
VAR_MASK_PATH = MODELS_DIR / "var_mask.npy"
VS_MODELS_PATH = MODELS_DIR / "vs_models.pkl"
DESCRIPTOR_NAMES_PATH = MODELS_DIR / "descriptor_names.npy"

MODEL_NAMES = ["XGBoost", "RF", "SVR", "GPR"]

# 训练集 T/P 覆盖范围（来自 data/zhao_data.csv 的 min/max），用于外推提示。
T_TRAIN_RANGE = (293.15, 403.15)
P_TRAIN_RANGE = (0.001, 96.3)

# 短名 → 全名映射（与根目录 predict.py 保持一致），便于下拉框使用简称。
CATION_SHORT = {
    "BMIM": "C4MIm or BMIM",
    "EMIM": "EMIM or C2MIm",
    "HMIM": "C6MIm or HMIM",
    "C8MIm": "C8MIm",
    "C2OHMIM": "C2OHMIM",
    "MEDAH": "MEDAH (CH2OH)2CH3NH",
    "DMEAH": "DMEAH (N11H,CH2OH)",
}
ANION_SHORT = {
    "BF4": "BF4",
    "PF6": "PF6",
    "TFSI": "TFSI or NTf2",
    "NTf2": "TFSI or NTf2",
    "TfO": "TfO OR TFMS",
    "OAc": "OAc or CH3COO",
    "EtSO4": "EtSO4",
    "MeSO4": "MeSO4",
    "L-lactate": "L-lactate",
    "CH3CH2CO2": "CH3CH2CO2",
    "TPTP": "(C2F5)3PF3 or TPTP",
    "HCO2": "HCO2",
}


def _runtime_descriptor_names() -> list[str]:
    """当前 RDKit 运行时生成的描述符顺序。"""
    return list(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles("C")).keys())


def _load_descriptor_names() -> list[str]:
    """加载训练阶段固定的描述符顺序锚点，并校验运行时顺序一致。

    锚点由 models/descriptor_names.npy 固化（217 个描述符名）。
    若锚点缺失（例如未随模型资产一起提供），回退到运行时顺序，仅保留数量检查。
    """
    runtime = _runtime_descriptor_names()
    if DESCRIPTOR_NAMES_PATH.exists():
        saved = np.load(DESCRIPTOR_NAMES_PATH, allow_pickle=True).tolist()
        if saved != runtime:
            raise RuntimeError(
                "RDKit 描述符顺序与训练阶段锚点不一致，拒绝推理以避免特征错位。"
                "请确认 RDKit 版本与训练阶段一致（requirements.txt 固定 2025.3.6）。"
            )
        return saved
    return runtime


# 描述符名称顺序：以锚点为准（与训练阶段 preprocess.compute_descriptors 及
# 根目录 predict.py 的取值顺序一致）。
_DESCRIPTOR_NAMES = _load_descriptor_names()

_assets = None
_measured_pairs = None


def _get_assets() -> dict:
    """惰性加载推理资产并做一致性校验。"""
    global _assets
    if _assets is not None:
        return _assets

    scaler = joblib.load(SCALER_PATH)
    nan_cols = np.load(NAN_COLS_PATH)
    var_mask = np.load(VAR_MASK_PATH)
    models = joblib.load(VS_MODELS_PATH)

    missing = [m for m in MODEL_NAMES if m not in models]
    if missing:
        raise KeyError(f"vs_models.pkl 缺少模型: {missing}")

    # 描述符名数量 ×2（阴阳离子）+ T + P 必须等于掩码长度。
    expected_raw = len(_DESCRIPTOR_NAMES) * 2 + 2
    if nan_cols.size != expected_raw:
        raise RuntimeError(
            f"描述符维度不匹配：{len(_DESCRIPTOR_NAMES)} 个描述符对应原始特征 "
            f"{expected_raw} 维，但 nan_cols_mask 为 {nan_cols.size} 维。"
        )
    if var_mask.size != int((~nan_cols).sum()):
        raise RuntimeError("var_mask 与 nan_cols_mask 长度不匹配。")

    _assets = {
        "models": models,
        "scaler": scaler,
        "nan_cols": nan_cols,
        "var_mask": var_mask,
    }
    return _assets


def _get_measured_pairs() -> set[tuple[str, str]]:
    """训练集中实际测量过的 (Cation, Anion) 组合（26 种）。"""
    global _measured_pairs
    if _measured_pairs is None:
        df = pd.read_csv(CLEAN_CSV)
        _measured_pairs = set(
            map(tuple, df[["Cation", "Anion"]].drop_duplicates().values)
        )
    return _measured_pairs


def _classify_ion(ion: str, smiles_map: dict, short_map: dict) -> tuple[str | None, str, bool]:
    """解析离子输入，返回 (标准全名, SMILES, 是否训练集已知)。

    名称优先，其次短名，最后视为原始 SMILES 并与训练集离子的 SMILES 精确匹配。
    """
    if ion in smiles_map:
        return ion, smiles_map[ion], True
    if ion in short_map:
        full = short_map[ion]
        return full, smiles_map[full], True
    for full, smi in smiles_map.items():
        if smi == ion:
            return full, ion, True
    return None, ion, False


def _descriptor_vector(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    desc = Descriptors.CalcMolDescriptors(mol)
    return np.array([desc[name] for name in _DESCRIPTOR_NAMES], dtype=float)


def build_feature(cation: str, anion: str, T: float, P: float) -> np.ndarray:
    """构造并标准化特征向量，返回 shape (1, n_features) 的数组。

    流程与训练阶段完全一致：
        阳离子描述符 + 阴离子描述符 + [T, P]
        -> nan_to_num -> ~nan_cols -> var_mask -> scaler.transform
    """
    assets = _get_assets()

    _, cat_smi, _ = _classify_ion(cation, cation_smiles, CATION_SHORT)
    _, ani_smi, _ = _classify_ion(anion, anion_smiles, ANION_SHORT)

    cat_vec = _descriptor_vector(cat_smi)
    ani_vec = _descriptor_vector(ani_smi)

    feat = np.concatenate([cat_vec, ani_vec, [float(T), float(P)]])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    feat = feat[~assets["nan_cols"]]
    feat = feat[assets["var_mask"]]

    if feat.size != assets["scaler"].n_features_in_:
        raise RuntimeError(
            f"筛选后特征维度 {feat.size} 与 scaler 期望的 "
            f"{assets['scaler'].n_features_in_} 不一致。"
        )
    return assets["scaler"].transform(feat.reshape(1, -1))


def known_status(cation: str, anion: str) -> dict:
    """判断离子液体的可靠分级。

    三档（区分「组分已知」与「完整组合已知」）：
    - 内插（组合已知）：该 (阳,阴) 组合在训练集中直接出现。
    - Tier 1（组分已知、组合新）：阴阳离子分别都在训练集出现，但组合未被测量。
    - 训练集外：至少一个离子未在训练集出现，无法按现有 Tier 规则可靠分级。

    V1.0 不做化学相似度算法。
    """
    cat_full, _, cat_known = _classify_ion(cation, cation_smiles, CATION_SHORT)
    ani_full, _, ani_known = _classify_ion(anion, anion_smiles, ANION_SHORT)

    pair_measured = False
    if cat_full is not None and ani_full is not None:
        pair_measured = (cat_full, ani_full) in _get_measured_pairs()

    if cat_known and ani_known and pair_measured:
        tier = "内插（组合已知）"
        note = "该离子液体组合在训练集中直接出现，预测属内插"
    elif cat_known and ani_known:
        tier = "Tier 1"
        note = "阴阳离子组分均在训练集出现，但该组合未被测量，预测属外推"
    else:
        tier = "训练集外"
        note = "含训练集未出现的离子，无法按现有 Tier 规则可靠分级，预测仅供优先级参考"

    return {
        "cation_known": cat_known,
        "anion_known": ani_known,
        "pair_measured": pair_measured,
        "tier": tier,
        "tier_note": note,
    }


# ============================================================
# 批量输入校验与风险判定
# ============================================================
def _validate_ion_field(label: str, ion, smi_map: dict, short_map: dict) -> list[str]:
    """校验单个离子字段，返回错误描述列表（合法返回空）。

    区分三类输入：
    - 缺失 / 为空 / 非字符串数值（NaN/Inf）
    - 非法 SMILES（无法被 RDKit 解析，且非已知名称/短名/精确 SMILES）
    - 未知但合法（可解析的 SMILES 或未知名称）——不在此处报错，仅由 known_status 标记
    """
    if ion is None:
        return [f"{label}缺失"]
    if isinstance(ion, str):
        s = ion.strip()
        if s == "":
            return [f"{label}为空"]
        if s in smi_map or s in short_map:
            return []
        for smi in smi_map.values():
            if smi == s:
                return []
        if Chem.MolFromSmiles(s) is None:
            return [f"{label}无法解析为合法 SMILES: {ion!r}"]
        return []
    # 非字符串输入：NaN/Inf 归为缺失，其余一律按类型错误拒绝
    try:
        if math.isnan(float(ion)) or math.isinf(float(ion)):
            return [f"{label}为 NaN/Inf，缺失"]
    except (TypeError, ValueError):
        pass
    return [f"{label}必须为字符串，当前类型 {type(ion).__name__}"]


def _validate_condition_field(label: str, val) -> list[str]:
    """校验温度/压力字段：必须为有限数值且 > 0。返回错误描述列表（合法返回空）。"""
    if val is None:
        return [f"{label}缺失"]
    if isinstance(val, bool):
        return [f"{label}为布尔值，非数值"]
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return [f"{label}为空"]
        try:
            x = float(s)
        except ValueError:
            return [f"{label}非数值: {val!r}"]
    else:
        try:
            x = float(val)
        except (TypeError, ValueError):
            return [f"{label}必须为数值，当前类型 {type(val).__name__}"]
    if not math.isfinite(x):
        return [f"{label}非有限数值（NaN/Inf）"]
    if x <= 0:
        return [f"{label}必须大于 0，当前 {x}"]
    return []


def validate_input(cation, anion, T, P) -> list[str]:
    """校验单条离子液体输入，返回错误描述列表（空列表表示全部通过）。"""
    errors = []
    errors += _validate_ion_field("阳离子", cation, cation_smiles, CATION_SHORT)
    errors += _validate_ion_field("阴离子", anion, anion_smiles, ANION_SHORT)
    errors += _validate_condition_field("温度 T", T)
    errors += _validate_condition_field("压力 P", P)
    return errors


def check_physical_range(result: dict) -> list[str]:
    """返回预测值落在摩尔分数物理范围 [0, 1] 之外的模型名列表（不裁剪）。"""
    return [name for name in MODEL_NAMES if not (0.0 <= result[name] <= 1.0)]


def risk_status(result: dict) -> str:
    """综合外推风险状态：物理范围异常 > 工况外推 > 正常。

    仅反映预测值物理合理性（[0,1]）与工况是否落在训练范围；
    离子化学空间覆盖由独立的 tier 字段表达，不并入本状态。
    """
    if check_physical_range(result):
        return "物理范围异常"
    if not result["T_in_training_range"] or not result["P_in_training_range"]:
        return "工况外推"
    return "正常"


def predict_il(cation: str, anion: str, T: float, P: float) -> dict:
    """预测单个离子液体在给定工况下的 H2S 溶解度（摩尔分数）。

    参数
    ----
    cation / anion : str
        阳离子、阴离子。可为全名、短名或原始 SMILES。
    T : float
        温度 (K)。
    P : float
        压力 (bar)。

    返回
    ----
    dict，含四模型各自预测值、mean/std/range，以及离子可靠分级、
    T、P 是否落在训练范围内等信息。
    """
    assets = _get_assets()
    feat_scaled = build_feature(cation, anion, T, P)

    preds = {
        name: float(assets["models"][name].predict(feat_scaled)[0])
        for name in MODEL_NAMES
    }

    vals = [preds[name] for name in MODEL_NAMES]
    result = {
        "cation": cation,
        "anion": anion,
        "T": float(T),
        "P": float(P),
        **preds,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "range": float(max(vals) - min(vals)),
        "T_in_training_range": T_TRAIN_RANGE[0] <= float(T) <= T_TRAIN_RANGE[1],
        "P_in_training_range": P_TRAIN_RANGE[0] <= float(P) <= P_TRAIN_RANGE[1],
    }
    result.update(known_status(cation, anion))
    return result


def predict_batch(records) -> tuple[list[dict], list[dict]]:
    """批量预测离子液体溶解度，逐行校验输入。

    参数
    ----
    records : 可迭代的 dict 序列，每项需含键 Cation / Anion / T / P。

    返回
    ----
    (results, errors)
    results : 成功预测的 dict 列表（含 tier、T/P 范围、risk_status、out_of_range_models）。
    errors  : 校验失败的 dict 列表，每项含 row(1 起)、原始 Cation/Anion/T/P、errors 列表。
    """
    results, errors = [], []
    for idx, rec in enumerate(records, start=1):
        cation = rec.get("Cation")
        anion = rec.get("Anion")
        T = rec.get("T")
        P = rec.get("P")

        errs = validate_input(cation, anion, T, P)
        if errs:
            errors.append({
                "row": idx,
                "Cation": cation,
                "Anion": anion,
                "T": T,
                "P": P,
                "errors": errs,
            })
            continue

        cation = cation.strip() if isinstance(cation, str) else cation
        anion = anion.strip() if isinstance(anion, str) else anion
        result = predict_il(cation, anion, float(T), float(P))
        result["out_of_range_models"] = check_physical_range(result)
        result["risk_status"] = risk_status(result)
        results.append(result)
    return results, errors
