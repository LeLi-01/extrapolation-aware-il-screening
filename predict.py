"""
预测脚本：加载 4 个预训练模型，输入 IL 结构 + 工况，输出预测结果。
用法：
  python predict.py "C4MIm or BMIM" "TFSI or NTf2" 303.15 1.0
  python predict.py BMIM TFSI 303.15 1.0          # 支持简称
  python predict.py --list                          # 列出可用离子
"""
import sys
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

# ══════════════════════════════════════════════════════════════
# SMILES 字典 (与 build_feature_matrix 保持一致)
# ══════════════════════════════════════════════════════════════
CATION_SMILES = {
    'C4MIm or BMIM':       'CCCC[n+]1ccn(C)c1',
    'EMIM or C2MIm':       'CC[n+]1ccn(C)c1',
    'C2OHMIM':             'OCC[n+]1ccn(C)c1',
    'C6MIm or HMIM':       'CCCCCC[n+]1ccn(C)c1',
    'C8MIm':               'CCCCCCCC[n+]1ccn(C)c1',
    'MEDAH (CH2OH)2CH3NH': 'C[NH+](CCO)CCO',
    'DMEAH (N11H,CH2OH)':  'C[NH+](C)CCO',
}

ANION_SMILES = {
    'BF4':                 '[B-](F)(F)(F)F',
    'EtSO4':               'CCOS(=O)(=O)[O-]',
    'MeSO4':               'COS(=O)(=O)[O-]',
    'PF6':                 'F[P-](F)(F)(F)(F)F',
    'TFSI or NTf2':        'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F',
    'TfO OR TFMS':         'O=S(=O)([O-])C(F)(F)F',
    'OAc or CH3COO':       'CC(=O)[O-]',
    'CH3CH2CO2':           'CCC(=O)[O-]',
    'L-lactate':           'C[C@@H](O)C(=O)[O-]',
    '(C2F5)3PF3 or TPTP':  'F[P-](F)(F)(C(F)(F)C(F)(F)F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F',
    'HCO2':                '[O-]C=O',
}

# 简称 → 全名映射
CATION_SHORT = {
    'BMIM': 'C4MIm or BMIM', 'EMIM': 'EMIM or C2MIm',
    'HMIM': 'C6MIm or HMIM', 'C8MIm': 'C8MIm',
    'C2OHMIM': 'C2OHMIM', 'MEDAH': 'MEDAH (CH2OH)2CH3NH',
    'DMEAH': 'DMEAH (N11H,CH2OH)',
}
ANION_SHORT = {
    'BF4': 'BF4', 'PF6': 'PF6', 'TFSI': 'TFSI or NTf2',
    'NTf2': 'TFSI or NTf2', 'TfO': 'TfO OR TFMS',
    'OAc': 'OAc or CH3COO', 'EtSO4': 'EtSO4', 'MeSO4': 'MeSO4',
    'L-lactate': 'L-lactate', 'CH3CH2CO2': 'CH3CH2CO2',
    'TPTP': '(C2F5)3PF3 or TPTP', 'HCO2': 'HCO2',
}


def _resolve(cation: str, anion: str):
    """将简称解析为 SMILES 字典的键名。"""
    cat_key = CATION_SHORT.get(cation, cation)
    ani_key = ANION_SHORT.get(anion, anion)
    if cat_key not in CATION_SMILES:
        raise KeyError(f"Unknown cation '{cation}'. Use --list to see options.")
    if ani_key not in ANION_SMILES:
        raise KeyError(f"Unknown anion '{anion}'. Use --list to see options.")
    return cat_key, ani_key


# ══════════════════════════════════════════════════════════════
# 一次性初始化 (模块加载时执行)
# ══════════════════════════════════════════════════════════════
print("Loading models...", end=" ", flush=True)

_scaler = joblib.load("scaler.pkl")
_models = {
    "XGBoost": joblib.load("model_XGBoost.pkl"),
    "RF":      joblib.load("model_RF.pkl"),
    "SVR":     joblib.load("model_SVR.pkl"),
    "GPR":     joblib.load("model_GPR.pkl"),
}
_nan_cols = np.load("nan_cols_mask.npy")
_var_mask = np.load("var_mask.npy")

# 预计算离子描述符 (7 cation + 11 anion)
_rdkit_desc_names = list(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles('C')).keys())

def _compute_descriptors(smiles_dict: dict) -> dict:
    """返回 {name: np.array(descriptors)}"""
    result = {}
    for name, smi in smiles_dict.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {name} → {smi}")
        desc = Descriptors.CalcMolDescriptors(mol)
        result[name] = np.array([desc[d] for d in _rdkit_desc_names])
    return result

_cation_desc = _compute_descriptors(CATION_SMILES)
_anion_desc  = _compute_descriptors(ANION_SMILES)

print("ready.")


# ══════════════════════════════════════════════════════════════
# 核心预测函数
# ══════════════════════════════════════════════════════════════
def predict(cation: str, anion: str, T: float, P: float) -> dict:
    """
    输入 IL 阳离子、阴离子名称 + 温度(K) + 压力(bar)，返回四模型预测。

    Parameters
    ----------
    cation : str
        阳离子名称 (全称或简称, 如 'BMIM' 或 'C4MIm or BMIM')
    anion : str
        阴离子名称 (全称或简称, 如 'TFSI' 或 'TFSI or NTf2')
    T : float
        温度 (K)
    P : float
        压力 (bar)

    Returns
    -------
    dict with keys: IL, T, P, XGBoost, RF, SVR, GPR, Mean, Std, Range
    """
    cat_key, ani_key = _resolve(cation, anion)
    il_name = f"[{cat_key}][{ani_key}]"

    # 拼接特征向量
    cat_vec = _cation_desc[cat_key]
    ani_vec = _anion_desc[ani_key]
    feat = np.concatenate([cat_vec, ani_vec, [T, P]])
    feat = np.nan_to_num(feat, nan=0.0)

    # 应用列掩码 (与训练时一致)
    feat = feat[~_nan_cols]
    feat = feat[_var_mask]

    # 标准化 + 预测
    feat_s = _scaler.transform(feat.reshape(1, -1))
    preds = {}
    for name, model in _models.items():
        preds[name] = float(model.predict(feat_s)[0])

    pvals = [preds[name] for name in ["XGBoost", "RF", "SVR", "GPR"]]
    preds["IL"] = il_name
    preds["T"] = T
    preds["P"] = P
    preds["Mean"] = round(np.mean(pvals), 4)
    preds["Std"] = round(np.std(pvals), 4)
    preds["Range"] = round(max(pvals) - min(pvals), 4)
    return preds


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def _print_result(r: dict):
    print(f"\n{'='*60}")
    print(f"  IL: {r['IL']}")
    print(f"  T = {r['T']} K,  P = {r['P']} bar")
    print(f"{'─'*60}")
    print(f"  H2S Solubility (mole fraction):")
    for m in ["XGBoost", "RF", "SVR", "GPR"]:
        print(f"    {m:<10} {r[m]:.4f}")
    print(f"{'─'*60}")
    print(f"  {'4-Model Mean':<18} {r['Mean']:.4f}")
    print(f"  {'Model Range':<18} {r['Range']:.4f}  (max − min)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        print("\nAvailable cations:")
        for short, full in CATION_SHORT.items():
            print(f"  {short:<10} → {full}")
        print("\nAvailable anions:")
        for short, full in ANION_SHORT.items():
            print(f"  {short:<10} → {full}")
        sys.exit(0)

    if len(sys.argv) == 5:
        # 命令行模式
        cation, anion = sys.argv[1], sys.argv[2]
        T = float(sys.argv[3])
        P = float(sys.argv[4])
        result = predict(cation, anion, T, P)
        _print_result(result)
    else:
        # 交互模式 (IDE 中直接运行)
        if len(sys.argv) > 1 and sys.argv[1] == "--list":
            print("\nAvailable cations:")
            for short, full in CATION_SHORT.items():
                print(f"  {short:<10} → {full}")
            print("\nAvailable anions:")
            for short, full in ANION_SHORT.items():
                print(f"  {short:<10} → {full}")
            sys.exit(0)

        print("\n" + "=" * 50)
        print("  H2S Solubility Prediction in Ionic Liquids")
        print("=" * 50)
        print("  Type cation/anion short name (e.g. BMIM, EMIM)")
        print("  Type 'list' to see all options, 'q' to quit")
        print("=" * 50)

        while True:
            cat = input("\n  Cation: ").strip()
            if cat.lower() == 'q':
                break
            if cat.lower() == 'list':
                print("  Cations: " + ", ".join(CATION_SHORT.keys()))
                print("  Anions:  " + ", ".join(ANION_SHORT.keys()))
                continue

            ani = input("  Anion:  ").strip()
            if ani.lower() == 'q':
                break
            if ani.lower() == 'list':
                print("  Cations: " + ", ".join(CATION_SHORT.keys()))
                print("  Anions:  " + ", ".join(ANION_SHORT.keys()))
                continue

            try:
                T = float(input("  T (K):  ").strip())
                P = float(input("  P (bar): ").strip())
                result = predict(cat, ani, T, P)
                _print_result(result)
            except KeyError as e:
                print(f"  Error: {e}")
            except ValueError:
                print("  Error: T and P must be numbers")
