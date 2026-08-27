import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path

# =========================
# 0. 项目路径
# =========================
# preprocess.py 位于 project/src/ 下，因此 parent.parent 是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_EXCEL = DATA_DIR / "Hydrogen sulfide solubility in ionic liquids (ILs).xlsx"
CLEAN_CSV = DATA_DIR / "zhao_data.csv"


#SMILES映射文件
cation_smiles = {
      'C4MIm or BMIM':       'CCCC[n+]1ccn(C)c1',
      'EMIM or C2MIm':       'CC[n+]1ccn(C)c1',
      'C2OHMIM':             'OCC[n+]1ccn(C)c1',
      'C6MIm or HMIM':       'CCCCCC[n+]1ccn(C)c1',
      'C8MIm':               'CCCCCCCC[n+]1ccn(C)c1',
      'MEDAH (CH2OH)2CH3NH': 'C[NH+](CCO)CCO',
      'DMEAH (N11H,CH2OH)':  'C[NH+](C)CCO',
  }

anion_smiles = {
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

# 根据论文Supporting Information和Results of ELM model表，以下ID未用于建模
removed_ids = [
    'C02011-A01001',   # [BMIM]Cl
    'C02011-A04003',   # [BMIM][TfO]
    'C03003-A03001',   # 1,2-二甲基-3-乙基咪唑双(三氟甲磺酰)亚胺
    'C11009-A03001',   # 丁基吡啶双(三氟甲磺酰)亚胺
    'C11067-A03001',   # 苄基吡啶双(三氟甲磺酰)亚胺
    'C11001-A03001',   # 丁基吡啶双(三氟甲磺酰)亚胺（另一种）
    'C08039-A03001',   # 苄基甲基吡咯烷双(三氟甲磺酰)亚胺
    'C03010-A03001',   # 丁基二甲基咪唑双(三氟甲磺酰)亚胺
    'C15164-A03001',   # N112,EtOH 双(三氟甲磺酰)亚胺（季铵类）
    'C15099-A03001',   # N114,EtOH 双(三氟甲磺酰)亚胺（季铵类）
    'C02003-A04003',   # [EMIM][TfO]（来自参考文献16，投稿时未发表）
    # 5月27日更新
    'C02011-A01002',  # [BMIM][Br]（仅1个数据点，无法参与LOIO-CV）
]

from sklearn.preprocessing import StandardScaler
def clean_raw_data(excel_path=RAW_EXCEL) -> pd.DataFrame:
    # 1. 读取数据
    df = pd.read_excel(excel_path, sheet_name="Database")
    #df 是一个内存中的“数据表格”
    # 2. 清洗列名（去掉所有空格）
    df.columns = df.columns.str.replace(' ', '', regex=False)

    # 3. 筛选目标列
    columns = ['No', 'ID', 'CID', 'Cation', 'AID', 'Anion', 'T(K)', 'P(bar)', 'Solubility', 'Method', 'Reference']
    df = df[columns]                #按列名读取

    # 4. 数据清洗
    df = df.dropna()                # 删除缺失值
    # df = df.drop_duplicates()       # 删除完全重复行
    # 5月24日更新
    df = df.drop_duplicates(subset=['ID', 'T(K)', 'P(bar)', 'Solubility'])   # 删除化学意义上完全重复的行
    # 5月27日更新
    df = df.drop_duplicates(subset=['ID', 'T(K)', 'P(bar)'], keep='first')  # 删除重复测量（如 Jou & M+ather343.15K / 8.0bar的repeat），保留首次记录值

    # # 5. 异常值检查（保留合理数据，仅打印警告）
    # def jinggao():
    #     print("T(K) < 0:", (df["T(K)"] < 0).sum())
    #     print("P(bar) < 0:", (df["P(bar)"] < 0).sum())
    #     print("Solubility < 0:", (df["Solubility"] < 0).sum())

    # ========== 新增：剔除未使用的11种离子液体 ==========
    # 删除 Reference 11 中不属于 Zhao Table 1 指定范围的 5 个冗余点
    # Zhao Table 1 仅对 [BMIM][Br] 纳入 Ref 48 数据，对其他 4 种 IL 排除
    ref11_exclude = df[
        (df['Reference'] == 11) &
        (df['ID'].isin(['C02011-A05001', 'C02011-A02001', 'C02011-A03001', 'C02003-A03001']))
        ].index
    df = df.drop(ref11_exclude)
    # 筛选：只保留 ID 不在 removed_ids 中的行
    df = df[~df['ID'].isin(removed_ids)]
    # print(f"剔除后剩余数据量：{len(df)} 条")
    # 6. 生成离子液体唯一标识
    df["IL_ID"] = df["CID"].astype(str) + "-" + df["AID"].astype(str)
    # 字符串拼接，用'+'。
    # 7. 保存结果
    # df.to_csv("zhao_data.csv", index=False)     #把 DataFrame 保存成 CSV 文件，并且不要把行索引写进去。
    # print(f"Saved {len(df)} rows to zhao_data.csv")
    df.to_csv("CLEAN_CSV", index=False)
    return df

# # 8. 输出统计信息
# print(f"不同 IL 种数: {df['IL_ID'].nunique()}")
# print(df["IL_ID"].value_counts())

# # 在清洗后，统计每个 IL 的数据点数
# def il_counts(df_check_counts: pd.DataFrame):
#     df_cleaned = df_check_counts[~df_check_counts['ID'].isin(removed_ids)]  # 您的清洗后数据
#     counts = df_cleaned['ID'].value_counts().sort_index()
#     return counts

# def df_linkid(df_link_ID: pd.DataFrame):
#     #为留一交叉验证做准备
#     df_link_ID["IL_ID"] = df_link_ID["CID"].astype(str) + "-" +df_link_ID["AID"].astype(str)
#     # n_unique = df_link_ID["IL_ID"].nunique()
#     # print(f"不同 IL 的种数：{n_unique}")
#     # print(df_link_ID["IL_ID"].value_counts()) #IL_ID的个数
#     return df_link_ID

#分子描述符计算
def il_sort_count(df):
    print("#分子描述符计算")
    unique_cations = df["Cation"].unique()
    unique_anions = df["Anion"].unique()
    print("Cations:", unique_cations) #7种
    print("Anions:", unique_anions) #12种
    return
    # 以上完成“读取和清洗数据”。

# ================= RDKit 严格验证 =================
def validate_smiles(smiles):
    if pd.isna(smiles):
        return False
    # RDKit 转换
    mol = Chem.MolFromSmiles(str(smiles))
    return mol is not None
# ================= 2. 读取与映射 =================
# 确保 zhao_data.csv 文件和你的代码在同一个文件夹下
# df = pd.read_csv('zhao_data.csv')

def add_smiles_yinshe_colums(df: pd.DataFrame):
    df['Cation_SMILES'] = df['Cation'].map(cation_smiles)
    df['Anion_SMILES'] = df['Anion'].map(anion_smiles)
    df['Cation_Valid'] = df['Cation_SMILES'].apply(validate_smiles)
    df['Anion_Valid'] = df['Anion_SMILES'].apply(validate_smiles)
    return df

# #查看RDkit版本
# import rdkit
# print("RDKit Version:", rdkit.__version__)

def compute_descriptors(smiles_dict: dict) -> pd.DataFrame:
    records = {}
    for name, smi in smiles_dict.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[警告] 无法解析 SMILES：{name} → {smi}")
            records[name] = {}
            continue
        records[name] = Descriptors.CalcMolDescriptors(mol)
    return pd.DataFrame.from_dict(records, orient="index")



# print("阳离子描述符 shape:", cation_df.shape) #(7, 217)  217与RDkit版本有关
# print("阴离子描述符 shape:", anion_df.shape)
# print(cation_df.head())

def link_xy(df_link_xy:pd.DataFrame,cation_df:pd.DataFrame,anion_df:pd.DataFrame):
    # ----------  拼接特征矩阵 X 和目标 y ----------
    X_list = []
    y_list = []

    for idx, row in df_link_xy.iterrows():
        cation = row["Cation"]
        anion = row["Anion"]
        # 取出描述符（转为 numpy 数组）
        cation_vec = cation_df.loc[cation].values
        anion_vec = anion_df.loc[anion].values
        T = row["T(K)"]
        P = row["P(bar)"]

        feat = np.concatenate([cation_vec, anion_vec, [T, P]])
        X_list.append(feat)
        y_list.append(row["Solubility"])

    X = np.array(X_list)  # (n, 436)
    y=np.array(y_list)
    # --- 删除含 NaN 的列 ---
    nan_cols = np.isnan(X).any(axis=0)  # 布尔数组，True = 该列有NaN；最终的返回值 nan_cols：它不再是一个矩阵，而是一个长度为 436 的一维布尔数组。
    X = X[:, ~nan_cols]  # 删除含NaN列
    # print(f"删除含NaN列: {nan_cols.sum()} 列, 剩余 {X.shape[1]} 列")

    # --- 删除零方差列 ---
    var_mask = X.var(axis=0) > 1e-6  # 布尔数组，True = 保留
    X = X[:, var_mask]
    # print(f"删除零方差列: {(var_mask == False).sum()} 列, 剩余 {X.shape[1]} 列")
    # --- 余下NaN填0（理论上此步后不应再有）---
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X,y,nan_cols,var_mask

def main():
    print("开始数据预处理...")

    # 创建目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # parents = True：自动创建缺失的级联父目录。如果你的路径定义是
    # data / processed /，但系统里连
    # data
    # 这个根文件夹都还没建，开启这个参数后，它会帮你把这一整串目录结构都建好。
    #
    # exist_ok = True：这是最核心的防御性参数。如果这个文件夹之前已经被创建过了，程序会自动跳过，绝对不会报错中断。
    df=clean_raw_data()

    #增加SMILES映射列
    df=add_smiles_yinshe_colums(df)

    print("开始验证 RDKit 兼容性...")
    invalid_cations = df[df['Cation_Valid'] == False][['Cation', 'Cation_SMILES']].drop_duplicates()
    invalid_anions = df[df['Anion_Valid'] == False][['Anion', 'Anion_SMILES']].drop_duplicates()
    print("\n" + "="*40)
    if invalid_cations.empty and invalid_anions.empty:
        print("所有 SMILES 全部通过 RDKit 严格验证")
    else:
        if not invalid_cations.empty:
            print(" 发现 RDKit 无法识别的阳离子：")
            print(invalid_cations)
        if not invalid_anions.empty:
            print("\n 发现 RDKit 无法识别的阴离子：")
            print(invalid_anions)
    #smiles转为mol
    cation_df = compute_descriptors(cation_smiles)
    anion_df = compute_descriptors(anion_smiles)

    print("阳离子描述符 shape:", cation_df.shape)  # (7, 217)  217与RDkit版本有关
    print("阴离子描述符 shape:", anion_df.shape)
    print(cation_df.head())

    #拼接特征矩阵与目标向量
    X, y, nan_cols, var_mask=link_xy(df,cation_df,anion_df)

    # --- 保存列掩码（后续SHAP特征名对齐用）---
    # np.save("nan_cols_mask.npy", nan_cols)
    # np.save("var_mask.npy", var_mask)
    # np.save("X_raw.npy", X)
    # np.save("y_raw.npy", y)
    np.save(MODELS_DIR / "nan_cols_mask.npy", nan_cols)
    np.save(MODELS_DIR / "var_mask.npy", var_mask)
    np.save(MODELS_DIR / "X_raw.npy", X)
    np.save(MODELS_DIR / "y_raw.npy", y)
    IL_ID=df["IL_ID"].values
    IL_ID=np.array(IL_ID,dtype=str)
    np.save(MODELS_DIR / "IL_ID.npy", IL_ID)

    np.save(MODELS_DIR /"IL_ID",IL_ID)


    print("预处理完成")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    main()