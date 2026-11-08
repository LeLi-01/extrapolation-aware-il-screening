"""
统一推理接口。

使用 models/ 目录下已持久化的四模型 (vs_models.pkl)、StandardScaler (scaler.pkl)
与两个特征掩码 (nan_cols_mask.npy / var_mask.npy) 进行推理，无需重新训练。

单位约定：
    T     —— 开尔文 (K)，训练范围 293.15–403.15
    P     —— bar，训练范围 0.001–96.3
    输出  —— H2S 溶解度（摩尔分数，无量纲）

注意：本模型预测的是 H2S 溶解度，仅作为筛选硫醇前置吸收离子液体的代理指标，
不代表硫醇溶解度本身。

设计要点：
    - 惰性加载资产，import 本模块无副作用（区别于根目录 predict.py 的模块级加载）。
    - 所有路径基于 __file__ 定位，不依赖当前工作目录。
    - 描述符按显式名称顺序取值，避免依赖 DataFrame 列序的隐式对齐。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.preprocess import anion_smiles, cation_smiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
NAN_COLS_PATH = MODELS_DIR / "nan_cols_mask.npy"
VAR_MASK_PATH = MODELS_DIR / "var_mask.npy"
VS_MODELS_PATH = MODELS_DIR / "vs_models.pkl"

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

_KNOWN_CATION_SMILES = set(cation_smiles.values())
_KNOWN_ANION_SMILES = set(anion_smiles.values())

# 描述符名称顺序：CalcMolDescriptors 的键序（RDKit 版本相关，requirements 固定 2025.3.6）。
# 与训练阶段 preprocess.compute_descriptors 及根目录 predict.py 的取值顺序一致。
_DESCRIPTOR_NAMES = list(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles("C")).keys())

_assets = None


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
            "请确认 RDKit 版本与训练阶段一致（requirements.txt 固定 2025.3.6）。"
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


def _resolve_ion(ion: str, smiles_map: dict, short_map: dict) -> tuple[str, bool]:
    """将离子输入解析为 SMILES。

    返回 (SMILES, 是否按已知名称命中)。名称优先，其次短名，否则视为原始 SMILES。
    """
    if ion in smiles_map:
        return smiles_map[ion], True
    if ion in short_map:
        return smiles_map[short_map[ion]], True
    return ion, False


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

    cat_smi, _ = _resolve_ion(cation, cation_smiles, CATION_SHORT)
    ani_smi, _ = _resolve_ion(anion, anion_smiles, ANION_SHORT)

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
    """判断阴阳离子是否属于训练集离子（按名称或精确 SMILES 匹配）。

    V1.0 不做化学相似度：只有「阴阳离子均为训练集出现」才视为可可靠分级（Tier 1），
    否则明确标记为训练集外、无法按现有 Tier 规则可靠分级。
    """
    cat_smi, cat_by_name = _resolve_ion(cation, cation_smiles, CATION_SHORT)
    ani_smi, ani_by_name = _resolve_ion(anion, anion_smiles, ANION_SHORT)

    cat_known = cat_by_name or (cat_smi in _KNOWN_CATION_SMILES)
    ani_known = ani_by_name or (ani_smi in _KNOWN_ANION_SMILES)

    if cat_known and ani_known:
        tier = "Tier 1"
        note = "阴阳离子均在训练集出现"
    else:
        tier = "训练集外"
        note = "含训练集未出现的离子，无法按现有 Tier 规则可靠分级，预测仅供优先级参考"

    return {
        "cation_known": cat_known,
        "anion_known": ani_known,
        "tier": tier,
        "tier_note": note,
    }


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
    dict，含四模型各自预测值、mean/std/range，以及离子是否已知 / Tier 分级 /
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
