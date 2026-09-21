import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path

# 项目路径
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
    'C02011-A01002',  # [BMIM][Br]（仅1个数据点，无法参与LOIO-CV）
]

from sklearn.preprocessing import StandardScaler
def clean_raw_data(excel_path=RAW_EXCEL) -> pd.DataFrame:
    #读取数据
    df = pd.read_excel(excel_path, sheet_name="Database")
    #清洗列名（去掉所有空格）
    df.columns = df.columns.str.replace(' ', '', regex=False)
    #筛选目标列
    columns = ['No', 'ID', 'CID', 'Cation', 'AID', 'Anion', 'T(K)', 'P(bar)', 'Solubility', 'Method', 'Reference']
    df = df[columns]                
    #数据清洗
    df = df.dropna()                
    # df = df.drop_duplicates()     
    df = df.drop_duplicates(subset=['ID', 'T(K)', 'P(bar)', 'Solubility'])   # 删除化学意义上完全重复的行
    df = df.drop_duplicates(subset=['ID', 'T(K)', 'P(bar)'], keep='first')  # 删除重复测量（如 Jou & M+ather343.15K / 8.0bar的repeat），保留首次记录值

    ref11_exclude = df[
        (df['Reference'] == 11) &
        (df['ID'].isin(['C02011-A05001', 'C02011-A02001', 'C02011-A03001', 'C02003-A03001']))
        ].index
    df = df.drop(ref11_exclude)
    #只保留ID不在removed_ids中的行
    df = df[~df['ID'].isin(removed_ids)]
    #生成离子液体唯一标识
    df["IL_ID"] = df["CID"].astype(str) + "-" + df["AID"].astype(str)
    df.to_csv("CLEAN_CSV", index=False)
    return df

#分子描述符计算
def il_sort_count(df):
    print("#分子描述符计算")
    unique_cations = df["Cation"].unique()
    unique_anions = df["Anion"].unique()
    print("Cations:", unique_cations) 
    print("Anions:", unique_anions) 
    return

#RDKit严格验证
def validate_smiles(smiles):
    if pd.isna(smiles):
        return False
    # RDKit 转换
    mol = Chem.MolFromSmiles(str(smiles))
    return mol is not None
      
#读取与映射 
def add_smiles_yinshe_colums(df: pd.DataFrame):
    df['Cation_SMILES'] = df['Cation'].map(cation_smiles)
    df['Anion_SMILES'] = df['Anion'].map(anion_smiles)
    df['Cation_Valid'] = df['Cation_SMILES'].apply(validate_smiles)
    df['Anion_Valid'] = df['Anion_SMILES'].apply(validate_smiles)
    return df

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

def link_xy(df_link_xy:pd.DataFrame,cation_df:pd.DataFrame,anion_df:pd.DataFrame):
    # 拼接特征矩阵 X 和目标 y
    X_list = []
    y_list = []
    for idx, row in df_link_xy.iterrows():
        cation = row["Cation"]
        anion = row["Anion"]
        cation_vec = cation_df.loc[cation].values
        anion_vec = anion_df.loc[anion].values
        T = row["T(K)"]
        P = row["P(bar)"]
        feat = np.concatenate([cation_vec, anion_vec, [T, P]])
        X_list.append(feat)
        y_list.append(row["Solubility"])
    X = np.array(X_list)  # (n, 436)
    y=np.array(y_list)
    nan_cols = np.isnan(X).any(axis=0)  
    X = X[:, ~nan_cols] 
    # 删除零方差列 ---
    var_mask = X.var(axis=0) > 1e-6  
    X = X[:, var_mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X,y,nan_cols,var_mask

def main():
    print("开始数据预处理...")
    # 创建目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df=clean_raw_data()
    #增加SMILES映射列
    df=add_smiles_yinshe_colums(df)
    print("开始验证 RDKit 兼容性...")
    invalid_cations = df[df['Cation_Valid'] == False][['Cation', 'Cation_SMILES']].drop_duplicates()
    invalid_anions = df[df['Anion_Valid'] == False][['Anion', 'Anion_SMILES']].drop_duplicates()
    print("\n" + "="*40)
    if invalid_cations.empty and invalid_anions.empty:
        print("所有SMILES全部通过RDKit严格验证")
    else:
        if not invalid_cations.empty:
            print("发现RDKit无法识别的阳离子：")
            print(invalid_cations)
        if not invalid_anions.empty:
            print("\n发现RDKit无法识别的阴离子：")
            print(invalid_anions)
    #smiles转为mol
    cation_df = compute_descriptors(cation_smiles)
    anion_df = compute_descriptors(anion_smiles)
    print("阳离子描述符shape:", cation_df.shape)  
    print("阴离子描述符shape:", anion_df.shape)
    print(cation_df.head())

    #拼接特征矩阵与目标向量
    X, y, nan_cols, var_mask=link_xy(df,cation_df,anion_df)
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
