import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
# 1. 读取数据
df = pd.read_excel(
    '/Users/lile/Downloads/Hydrogen sulfide solubility in ionic liquids (ILs).xlsx',
    sheet_name="Database"
)
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

# 5. 异常值检查（保留合理数据，仅打印警告）
print("T(K) < 0:", (df["T(K)"] < 0).sum())
print("P(bar) < 0:", (df["P(bar)"] < 0).sum())
print("Solubility < 0:", (df["Solubility"] < 0).sum())
#待检查
# ========== 新增：剔除未使用的11种离子液体 ==========
# 删除 Reference 11 中不属于 Zhao Table 1 指定范围的 5 个冗余点
# Zhao Table 1 仅对 [BMIM][Br] 纳入 Ref 48 数据，对其他 4 种 IL 排除
ref11_exclude = df[
    (df['Reference'] == 11) &
    (df['ID'].isin(['C02011-A05001', 'C02011-A02001', 'C02011-A03001', 'C02003-A03001']))
    ].index
df = df.drop(ref11_exclude)
# 5月24日更新

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

print("原始唯一IL种数:", df['ID'].nunique())

# 筛选：只保留 ID 不在 removed_ids 中的行
df = df[~df['ID'].isin(removed_ids)]

print(f"剔除后剩余数据量：{len(df)} 条")
# 6. 生成离子液体唯一标识
df["IL_ID"] = df["CID"].astype(str) + "-" + df["AID"].astype(str)
# 字符串拼接，用'+'。

# 7. 保存结果
df.to_csv("zhao_data.csv", index=False)     #把 DataFrame 保存成 CSV 文件，并且不要把行索引写进去。
print(f"Saved {len(df)} rows to zhao_data.csv")

# # 8. 输出统计信息
# print(f"不同 IL 种数: {df['IL_ID'].nunique()}")
# print(df["IL_ID"].value_counts())

# 在清洗后，统计每个 IL 的数据点数
df_cleaned = df[~df['ID'].isin(removed_ids)]  # 您的清洗后数据
counts = df_cleaned['ID'].value_counts().sort_index()
print(counts)


#为留一交叉验证做准备
df = pd.read_csv("zhao_data.csv")

df["IL_ID"] = df["CID"].astype(str) + "-" + df["AID"].astype(str)

n_unique = df["IL_ID"].nunique()
print(f"不同 IL 的种数：{n_unique}")
print(df["IL_ID"].value_counts()) #IL_ID的个数

# 如果
# CID / AID
# 是整数，拼接前需要格式化为零填充字符串（如
# C02011），则改为：
#
# df["IL_ID"] = df["CID"].apply(lambda x: f"C{int(x):05d}") + "-" + df["AID"].apply(lambda x:f"A{int(x):05d}")

# 具体格式取决于原始数据中CID / AID 的实际值，建议先
# print(df[["CID", "AID"]].head())
# 确认格式再选用哪种拼接方式。

#分子描述符计算
print("#分子描述符计算")
df = pd.read_csv("zhao_data.csv")
unique_cations = df["Cation"].unique()
unique_anions = df["Anion"].unique()
print("Cations:", unique_cations) #7种
print("Anions:", unique_anions) #12种
# 以上完成“读取和清洗数据”。


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
#Claude   建议用 rdkit 的 Chem.MolFromSmiles() 逐条验证，返回 None 则表示 SMILES 有语法错误
#我： 不是的。RDKit 并不是在和一个“内置的数据库或字典”进行比对；
# rdkit 的 Chem.MolFromSmiles()这个是rdkit中自带的包，我们是在与包中的数据进行核对？
#它其实是在做“化学语法和物理化学常识的实时检查”；
#5月3日
#
# ================= 2. 读取与映射 =================
print("正在读取数据并映射 SMILES...")
# 确保 zhao_data.csv 文件和你的代码在同一个文件夹下
df = pd.read_csv('zhao_data.csv')
df['Cation_SMILES'] = df['Cation'].map(cation_smiles)
df['Anion_SMILES'] = df['Anion'].map(anion_smiles)

# ================= 3. RDKit 严格验证 =================
def validate_smiles(smiles):
    if pd.isna(smiles):
        return False
    # RDKit 转换
    mol = Chem.MolFromSmiles(str(smiles))
    return mol is not None

print("开始验证 RDKit 兼容性...")
df['Cation_Valid'] = df['Cation_SMILES'].apply(validate_smiles)
df['Anion_Valid'] = df['Anion_SMILES'].apply(validate_smiles)

invalid_cations = df[df['Cation_Valid'] == False][['Cation', 'Cation_SMILES']].drop_duplicates()
invalid_anions = df[df['Anion_Valid'] == False][['Anion', 'Anion_SMILES']].drop_duplicates()

print("\n" + "="*40)
if invalid_cations.empty and invalid_anions.empty:
    print("太棒了！所有 SMILES 全部通过 RDKit 严格验证！")
else:
    if not invalid_cations.empty:
        print(" 发现 RDKit 无法识别的阳离子：")
        print(invalid_cations)
    if not invalid_anions.empty:
        print("\n 发现 RDKit 无法识别的阴离子：")
        print(invalid_anions)
print("="*40)
#查看RDkit版本
import rdkit
print("RDKit Version:", rdkit.__version__)

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


cation_df = compute_descriptors(cation_smiles)
anion_df = compute_descriptors(anion_smiles)

print("阳离子描述符 shape:", cation_df.shape) #(7, 217)  217与RDkit版本有关
print("阴离子描述符 shape:", anion_df.shape)
print(cation_df.head())

# 说明：
#
# - Descriptors.descList是RDKit所有2D描述符的列表（通常~200个），每项为(名称, 函数)元组，无需手动枚举。- 解析失败的离子会打印警告并填
# None，便于排查SMILES错误。- cation_df和anion_df可直接pd.concat或与溶解度数据merge使用。
from sklearn.preprocessing import StandardScaler

# 将描述符合并到原始数据
# ---------- 1. 读取数据 ----------
df = pd.read_csv("zhao_data.csv")  # 你的清洗后数据

# ---------- 2. 确保描述符 DataFrame 的索引名称与 df 中的离子名称完全一致 ----------
# 你的 cation_df 和 anion_df 已经构建好，索引就是离子名称（如 'C4MIm or BMIM'）

# ---------- 3. 拼接特征矩阵 X 和目标 y ----------
X_list = []
y_list = []

for idx, row in df.iterrows():
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
#
# 5月27日
# X = np.array(X_list)
# #日期5月14日
# # NaN/Inf 清理必须在保存之前
# X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

X = np.array(X_list)  # (n, 436)

# --- 删除含 NaN 的列 ---
nan_cols = np.isnan(X).any(axis=0)  # 布尔数组，True = 该列有NaN；最终的返回值 nan_cols：它不再是一个矩阵，而是一个长度为 436 的一维布尔数组。
X = X[:, ~nan_cols]  # 删除含NaN列
print(f"删除含NaN列: {nan_cols.sum()} 列, 剩余 {X.shape[1]} 列")

# --- 删除零方差列 ---
var_mask = X.var(axis=0) > 1e-6  # 布尔数组，True = 保留
X = X[:, var_mask]
print(f"删除零方差列: {(var_mask == False).sum()} 列, 剩余 {X.shape[1]} 列")

# --- 余下NaN填0（理论上此步后不应再有）---
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# --- 保存列掩码（后续SHAP特征名对齐用）---
np.save("nan_cols_mask.npy", nan_cols)
np.save("var_mask.npy", var_mask)

np.save("X_raw.npy", X)

# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)
# # np.save("X_scaled.npy", X_scaled)  # 现在保存的是干净数据
#
# y = np.array(y_list)
#
# # print(f"特征矩阵形状: {X.shape}")  # 应为 (样本数, 阳离子描述符数+阴离子描述符数+2)
# # 保存未标准化的原始特征矩阵，用于 LOIO-CV
# # np.save("X_raw.npy", X)
# # 5月27日：位置1已完成NaN列删除+零方差列删除，此处重复代码已移除
# # ---------- 4. 标准化（仅用于最终模型和 SHAP）----------
# # 8月7日 下面两段重复了
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)
#
# # ---------- 5. 保存结果 ----------
# # np.save("X_scaled.npy", X_scaled)
# np.save("y.npy", y)
# import joblib
# joblib.dump(scaler, "scaler.pkl")
# print("特征矩阵和标签已保存。")
#
#
#
# # Python 导入标准库 os 模块的语句。
# # 这个模块让你能够用 Python 代码直接调用操作系统的功能，比如文件路径操作、环境变量、执行系统命令等，而不需要手动去点鼠标。
# import os
# # 这是在检查文件 X_scaled.npy 是否存在于当前目录，返回 True 或 False。
# print(os.path.exists("X_scaled.npy"))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 1. 加载数据（建议从保存的 il_ids.npy 读取，避免顺序风险）
# 日期8月7日。又定义了一遍df，好像有问题
df = pd.read_csv("zhao_data.csv")
X_raw = np.load("X_raw.npy")
y = np.load("y.npy")
IL_IDs = df["IL_ID"].values   # 如果之前保存了 il_ids.npy，则优先使用 np.load

# 验证顺序一致性（可选）
print(f"X_raw shape: {X_raw.shape}, y shape: {y.shape}, IL_IDs length: {len(IL_IDs)}")

# 2. 处理 NaN / Inf
X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

# 3. 确定所有 IL 种类
unique_ILs = np.unique(IL_IDs)
print(f"总样本数: {len(y)}, 原始特征维度: {X_raw.shape[1]}, IL 种类: {len(unique_ILs)}")

# 5月28日：LOIO-CV 已重构为多模型函数，见下方

# # 5. LOIO-CV 主循环

# --- 5月28日：重构为函数，支持多模型对比 ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
# 虽然它们都在 sklearn.gaussian_process 这个“大柜子”里，但 Python 的导入机制不会因为导入了】
# GaussianProcessRegressor 就自动把所有核函数也带进来。
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
# 拿 随机划分函数，用来生成训练/测试集，评估模型在随机采样下的表现。
from sklearn.model_selection import train_test_split

def _make_gpr():
    """GPR每折需重新实例化（fit后内核状态不可重用）
    RBF length_scale=10.0适配~200维特征空间（典型点间距~20）
    WhiteKernel noise=0.1（信噪比≈10），避免noise=1.0导致的过度平滑"""
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(0.1),
        alpha=1e-5, normalize_y=True, random_state=42,
        optimizer=None,
    )

# ==========================================
# 随机 80/20 划分内部基线
# ==========================================
def run_random_split_baseline(model, model_name, X_raw, y, n_repeats=10):
    """随机 80/20 划分，重复 n_repeats 次"""
    r2_list, rmse_list, mae_list = [], [], []
    for i in range(n_repeats):
        # 随机划分，标准写法
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.20, random_state=i * 42
        )
        scaler_rs = StandardScaler()
        X_train_scaled = scaler_rs.fit_transform(X_train)
        X_test_scaled = scaler_rs.transform(X_test)
        if model_name == "GPR":
            fold_m = _make_gpr()
        else:
            fold_m = model
        fold_m.fit(X_train_scaled, y_train)
        y_pred_rs = fold_m.predict(X_test_scaled)
        r2_list.append(r2_score(y_test, y_pred_rs))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred_rs)))
        mae_list.append(mean_absolute_error(y_test, y_pred_rs))
        print(f"  {model_name} Repeat {i+1:02d} | R²: {r2_list[-1]:.4f} | RMSE: {rmse_list[-1]:.4f} | MAE: {mae_list[-1]:.4f}")
    return {
        "model": model_name,
        "R2_mean": np.mean(r2_list), "R2_std": np.std(r2_list),
        "RMSE_mean": np.mean(rmse_list), "RMSE_std": np.std(rmse_list),
        "MAE_mean": np.mean(mae_list), "MAE_std": np.std(mae_list),
    }

random_models = {
    "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0),
    "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.01),
    "GPR": _make_gpr(),
}

print("\n" + "="*50)
print("=== 随机划分 80/20 多模型对比 ===")
print("="*50)
random_results_list = []
for name, model in random_models.items():
    print(f"\n--- 随机划分基线: {name} ---")
    res = run_random_split_baseline(model, name, X_raw, y)
    random_results_list.append(res)
    print(f"  {name} 平均 R²: {res['R2_mean']:.4f} ± {res['R2_std']:.4f}")

random_comparison = pd.DataFrame(random_results_list)
print("\n" + random_comparison.to_string(index=False))
random_comparison.to_csv("random_split_model_comparison.csv", index=False)
# 这是 Pandas DataFrame 自带的一个方法。它不直接打印，而是把表格里的数据转换成一段非常规整的纯文本字符串。
print("已保存: random_split_model_comparison.csv")

def run_loio_cv(model, model_name, unique_ILs, IL_IDs, X_raw, y):
    """对指定模型执行 LOIO-CV，返回 (全局指标dict, 每折results_df, 全局y_true, 全局y_pred)"""
    global_y_true = []
    global_y_pred = []
    fold_results = []

    for test_il in unique_ILs:
        test_idx = (IL_IDs == test_il)
        train_idx = ~test_idx

        X_train = X_raw[train_idx]
        X_test = X_raw[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5月28日：GPR 每折需重新实例化，其余模型 fit() 即可重置
        if model_name == "GPR":
            fold_model = _make_gpr()
        else:
            fold_model = model

        fold_model.fit(X_train_scaled, y_train)
        y_pred = fold_model.predict(X_test_scaled)

        global_y_true.extend(y_test)             #列表，才有.extend()
        global_y_pred.extend(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        aard = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
        r2_fold = r2_score(y_test, y_pred) if len(y_test) >= 2 else float('nan')

        fold_results.append({
            "IL_ID": test_il,
            "R2": r2_fold,
            "RMSE": rmse,
            "MAE": mae,
            "AARD": aard,
            "n_test": len(y_test)
        })
    # 转换为 NumPy 类型，是为了“运行速度”和“代码健壮性”，而不是为了“满足 r2_score 的门槛”。
    global_y_true = np.array(global_y_true)
    global_y_pred = np.array(global_y_pred)

    global_metrics = {
        "model": model_name,
        "R2": r2_score(global_y_true, global_y_pred),
        "RMSE": np.sqrt(mean_squared_error(global_y_true, global_y_pred)),
        "MAE": mean_absolute_error(global_y_true, global_y_pred),
        "n_total": len(global_y_true)
    }
    return global_metrics, pd.DataFrame(fold_results), global_y_true, global_y_pred


loio_models = {
    "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0),
    "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.01),
    "GPR": _make_gpr(),  # 5月28日：新增GPR
}

loio_global_results = []
loio_fold_results = {}
loio_predictions = {}

for name, model in loio_models.items():
    print(f"\n{'='*40}")
    print(f"开始执行 LOIO-CV: {name}")
    print(f"{'='*40}")
    global_m, fold_df, yt, yp = run_loio_cv(model, name, unique_ILs, IL_IDs, X_raw, y)
    loio_global_results.append(global_m)
    loio_fold_results[name] = fold_df
    loio_predictions[name] = (yt, yp)
    # XGBoost 打印单折（论文表4需要），RF 仅汇总
    if name == "XGBoost":
        for _, row in fold_df.iterrows():
            #_：这是一个约定俗成的变量名，表示“我不关心这个值”。在这里代表行索引（0, 1, 2...），
            # 因为你不需要它，所以用 _ 丢弃了。
            print(f"  {row['IL_ID']:<15} | R²: {row['R2']:.4f} | RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f} | n: {int(row['n_test'])}")
    print(f"\n=== LOIO-CV ({name}) 全局评估 ===")
    print(f"总测试样本数 : {global_m['n_total']}")
    print(f"全局 R²      : {global_m['R2']:.4f}")
    print(f"全局 RMSE    : {global_m['RMSE']:.4f}")
    print(f"全局 MAE     : {global_m['MAE']:.4f}")

# 保存四种模型每折结果
for name, fold_df in loio_fold_results.items():
    fold_df.to_csv(f"loio_cv_{name}_per_il.csv", index=False)
print("\n已保存四种模型逐 IL 结果: loio_cv_*_per_il.csv")

# 保留 XGBoost 结果给后续分析
results_df = loio_fold_results["XGBoost"]

# --- 5月28日：多模型 LOIO-CV 对比汇总 ---
print("\n" + "=" * 50)
print("=== LOIO-CV 多模型对比汇总 ===")
loio_comparison = pd.DataFrame(loio_global_results)
print(loio_comparison.to_string(index=False))
loio_comparison.to_csv("loio_cv_model_comparison.csv", index=False)
print("已保存: loio_cv_model_comparison.csv")
# --- LOIO-CV vs 随机划分 综合对比表 ---
print("\n" + "="*60)
print("=== 综合对比：LOIO-CV vs 随机划分（多模型）===")
summary_rows = []
for loio_row in loio_global_results:
    name = loio_row["model"]
    rand_row = next(r for r in random_results_list if r["model"] == name)
    summary_rows.append({
        "模型": name,
        "随机划分 R²": f"{rand_row['R2_mean']:.4f} ± {rand_row['R2_std']:.4f}",
        "LOIO-CV R²": f"{loio_row['R2']:.4f}",
        "R² 落差": f"{rand_row['R2_mean'] - loio_row['R2']:.4f}",
        "随机划分 RMSE": f"{rand_row['RMSE_mean']:.4f}",
        "LOIO-CV RMSE": f"{loio_row['RMSE']:.4f}",
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("loio_vs_random_comparison.csv", index=False)
print("\n已保存: loio_vs_random_comparison.csv")

# --- 5月28日：多维度对比表（含逐IL RMSE分布、R²>0.5占比等）---
print("\n" + "="*80)
print("=== 多维度综合对比 ===")
print("="*80)
multi_rows = []
for name, fold_df in loio_fold_results.items():
    loio_row = next(r for r in loio_global_results if r["model"] == name)
    rand_row = next(r for r in random_results_list if r["model"] == name)
    valid = fold_df[fold_df['R2'].notna()]
    n_total = len(fold_df)
    multi_rows.append({
        "模型": name,
        "随机R²": f"{rand_row['R2_mean']:.4f}±{rand_row['R2_std']:.4f}",
        "LOIO R²": f"{loio_row['R2']:.4f}",
        "ΔR²": f"{rand_row['R2_mean'] - loio_row['R2']:.4f}",
        "LOIO RMSE": f"{loio_row['RMSE']:.4f}",
        "LOIO MAE": f"{loio_row['MAE']:.4f}",
        "ΔRMSE": f"{loio_row['RMSE'] - rand_row['RMSE_mean']:.4f}",
        "IL_RMSE中位数": f"{fold_df['RMSE'].median():.4f}",
        "IL_RMSE_IQR": f"{fold_df['RMSE'].quantile(0.75) - fold_df['RMSE'].quantile(0.25):.4f}",
        "R²>0.5占比": f"{(valid['R2'] > 0.5).sum()}/{n_total} ({(valid['R2'] > 0.5).sum()/n_total:.1%})",
        "R²>0.8占比": f"{(valid['R2'] > 0.8).sum()}/{n_total} ({(valid['R2'] > 0.8).sum()/n_total:.1%})",
        "R²<0占比": f"{(valid['R2'] < 0).sum()}/{n_total} ({(valid['R2'] < 0).sum()/n_total:.1%})",
    })
multi_df = pd.DataFrame(multi_rows)
print(multi_df.to_string(index=False))
multi_df.to_csv("multimetric_comparison.csv", index=False)
print("\n已保存: multimetric_comparison.csv")
print("="*60)

# ==========================================
# 提取并分析预测误差最大的 离子液体 (ILs)
# ==========================================
print("\n" + "="*50)
print("开始深挖：LOIO-CV 预测最差的离子液体 Top 10")

# 5月28日：results_df 已在 LOIO-CV 部分定义（XGBoost 每折结果），此处复用
# 按照 RMSE 降序排列（误差越大的排在越前面）
worst_ils = results_df.sort_values(by="RMSE", ascending=False).head(10)

print(worst_ils.to_string(index=False))
print("="*50)

# ==========================================
# 提取 Top 10 误差 IL 的真实化学名称
# ==========================================

print("\n" + "="*50)
print("正在解码 Top 10 误差分子的真实化学结构...")

# 读取原始数据以获取化学名称映射
df_names = pd.read_csv("zhao_data.csv")

# 提取黑名单里的 IL_ID
worst_id_list = worst_ils['IL_ID'].tolist()

# 过滤出这些 ID 对应的阳离子和阴离子名字（去重）
worst_names_df = df_names[df_names['IL_ID'].isin(worst_id_list)][['IL_ID', 'Cation', 'Anion']].drop_duplicates()

# 将化学名字合并到误差结果表中
final_worst_table = pd.merge(worst_ils, worst_names_df, on='IL_ID', how='left')

# 打印最终结果（按 RMSE 降序）
print(final_worst_table[['IL_ID', 'Cation', 'Anion', 'RMSE', 'MAE', 'n_test']].to_string(index=False))
print("="*50)

import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
import warnings

warnings.filterwarnings('ignore')

# ── 字体: Times New Roman, 数学符号 STIX ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 10,
    'mathtext.fontset': 'stix',
    'svg.fonttype': 'none',
})

# ==========================================
# 1. 自动重构特征名称列表 (极其关键，写论文必备)
# ==========================================
print("正在重构特征名称列表...")
# 获取 RDKit 的所有 2D 描述符名称
rdkit_desc_names = list(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles('C')).keys())

# 5月27日：使用位置1保存的掩码对齐特征名（先删NaN列，再删零方差列）
all_feature_names = [f"Cat_{name}" for name in rdkit_desc_names] + \
                [f"Ani_{name}" for name in rdkit_desc_names] + \
                ["T(K)", "P(bar)"]

nan_cols = np.load("nan_cols_mask.npy")
var_mask_saved = np.load("var_mask.npy")
valid_feature_names = np.array(all_feature_names)[~nan_cols][var_mask_saved].tolist()

# ==========================================
# 2. 加载全量标准化数据并训练最终模型
# ==========================================
print("加载数据并训练全量 XGBoost 模型...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
np.save("X_scaled.npy", X_scaled)  # 现在保存的是干净数据
y = np.array(y_list)
np.save("y.npy", y)
import joblib
joblib.dump(scaler, "scaler.pkl")
print("特征矩阵和标签已保存。")
# Python 导入标准库 os 模块的语句。
# 这个模块让你能够用 Python 代码直接调用操作系统的功能，比如文件路径操作、环境变量、执行系统命令等，而不需要手动去点鼠标。
import os
# 这是在检查文件 X_scaled.npy 是否存在于当前目录，返回 True 或 False。
print(os.path.exists("X_scaled.npy"))
X_scaled = np.load("X_scaled.npy")
y = np.load("y.npy")

# 5月27日：X_scaled已在构建阶段完成列过滤，直接使用
X_filtered = X_scaled

# 训练最终模型 (复用你验证过的超参数)
final_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
final_model.fit(X_filtered, y)
print("模型训练完成！")
# 5月14日
# ================== 使用 joblib 保存模型 ==================
# 5月27日：var_mask.npy 已在特征构建阶段保存，此处不再重复
import joblib # 确保顶部有 import joblib

joblib.dump(final_model, 'final_model.pkl') # 换成 pkl 格式
print("模型已成功使用 joblib 保存！")
# ===============================================================

# ==========================================
# 3. SHAP 解释器初始化
# ==========================================
print("正在计算 SHAP 值 (可能需要几秒到十几秒)...")
# 使用 TreeExplainer，它是专门针对树模型（如 XGBoost）的高效解释器
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_filtered)

# ==========================================
# 4. 绘图 1：全局特征重要性条形图 (Top 20)
# ==========================================
print("生成全局特征重要性条形图...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_filtered,
    feature_names=valid_feature_names,
    plot_type="bar",
    max_display=20,  # 只显示最重要的 20 个特征
    show=False
)
plt.title("Global Feature Importance (SHAP Bar Plot)", fontweight='bold')
plt.tight_layout()
plt.savefig("shap_bar_importance.png", dpi=300)
plt.show()

# ==========================================
# 5. 绘图 2：SHAP 摘要散点图 (既看重要性，又看正负影响，强烈推荐！)
# ==========================================
print("生成 SHAP 摘要散点图...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_filtered,
    feature_names=valid_feature_names,
    max_display=20,
    show=False
)
plt.title("SHAP Summary Plot (Impact Direction)", fontweight='bold')
plt.tight_layout()
plt.savefig("shap_summary_scatter.png", dpi=300)
plt.show()

# ==========================================
# 6. 绘图 3：关键变量的依赖图 (Dependence Plots)
# ==========================================
print("生成关键变量依赖图...")
# 我们强制画出压力 P(bar) 和 温度 T(K) 的依赖图，因为它们是明确的热力学条件
# 如果 T(K) 或 P(bar) 因为方差过滤被去掉了（不太可能），这里会报错，所以加个安全获取索引的逻辑
try:
    p_idx = valid_feature_names.index("P(bar)")
    plt.figure()
    shap.dependence_plot(
        p_idx,
        shap_values,
        X_filtered,
        feature_names=valid_feature_names,
        show=False
    )
    plt.title("SHAP Dependence Plot: P(bar)", fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_dependence_P.png", dpi=300)
    plt.show()

    t_idx = valid_feature_names.index("T(K)")
    plt.figure()
    shap.dependence_plot(
        t_idx,
        shap_values,
        X_filtered,
        feature_names=valid_feature_names,
        show=False
    )
    plt.title("SHAP Dependence Plot: T(K)", fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_dependence_T.png", dpi=300)
    plt.show()

except ValueError as e:
    print(f"未能找到 T(K) 或 P(bar) 特征，可能拼写不一致或被过滤：{e}")

print("所有 SHAP 分析图表已生成并保存！")

# =========================================================================
# 虚拟筛选：全组合扫描，四模型共识预测 + 分级置信度
# =========================================================================
print("\n" + "=" * 80)
print("=== 虚拟筛选：全组合扫描 ===")
print("=" * 80)

# ========== 1. 加载预处理对象 ==========
scaler = joblib.load('scaler.pkl')
var_mask = np.load('var_mask.npy')
nan_cols = np.load("nan_cols_mask.npy")

# ========== 2. 构建筛选池 ==========
# 阳离子简称映射
cation_short = {
    'C4MIm or BMIM': 'BMIM',
    'EMIM or C2MIm': 'EMIM',
    'C2OHMIM': 'C2OHMIM',
    'C6MIm or HMIM': 'HMIM',
    'C8MIm': 'C8MIm',
    'MEDAH (CH2OH)2CH3NH': 'MEDAH',
    'DMEAH (N11H,CH2OH)': 'DMEAH',
}
# 阴离子简称映射
anion_short = {
    'BF4': 'BF4', 'PF6': 'PF6', 'TFSI or NTf2': 'TFSI',
    'TfO OR TFMS': 'TfO', 'OAc or CH3COO': 'OAc',
    'CH3CH2CO2': 'CH3CH2CO2', 'L-lactate': 'L-lactate',
    '(C2F5)3PF3 or TPTP': 'TPTP', 'HCO2': 'HCO2',
    'EtSO4': 'EtSO4', 'MeSO4': 'MeSO4',
}

train_cations = list(cation_smiles.keys())   # 7
train_anions = list(anion_smiles.keys())      # 11
measured_pairs = set()
for _, row in df[['Cation', 'Anion']].drop_duplicates().iterrows():
    measured_pairs.add((row['Cation'], row['Anion']))

# Tier 1: 已知离子新组合（7x11 - 26 = 51）
tier1_pool = []
for cat in train_cations:
    for ani in train_anions:
        if (cat, ani) not in measured_pairs:
            tier1_pool.append((cat, ani, 1))

# Tier 2-4: 新阴离子（TFA/SCN/DCN）x 已知阳离子 = 21
new_anion_smiles = {
    'TFA': '[O-]C(=O)C(F)(F)F',
    'SCN': '[S-]C#N',
    'DCN': 'N#C[N-]C#N',
}
# TFA: 羧酸根家族（类似 OAc, L-lactate, CH3CH2CO2）-> Tier 2
# SCN: 拟卤素，仅与 TFSI/TfO 共享 S 原子 -> Tier 3
# DCN: 二氰胺，训练集无氰基阴离子 -> Tier 4
new_anion_tier = {'TFA': 2, 'SCN': 3, 'DCN': 4}
tier234_pool = []
for ani_name, ani_smi in new_anion_smiles.items():
    for cat in train_cations:
        tier234_pool.append((cat, ani_name, new_anion_tier[ani_name]))

all_pool = tier1_pool + tier234_pool  # 51 + 21 = 72
print(f"筛选池: {len(tier1_pool)} Tier 1 (已知离子新组合) + {len(tier234_pool)} Tier 2-4 (新阴离子) = {len(all_pool)} ILs")

# 合并所有 SMILES 字典
all_cation_smiles = cation_smiles
all_anion_smiles = {**anion_smiles, **new_anion_smiles}
# 扩展阴离子简称映射
anion_short.update({'TFA': 'TFA', 'SCN': 'SCN', 'DCN': 'DCN'})

# ========== 3. 计算描述符 ==========
cation_desc = compute_descriptors(all_cation_smiles)
anion_desc = compute_descriptors(all_anion_smiles)

# ========== 4. 训练四模型（全量数据）==========
print("\n正在为虚拟筛选训练四模型（全量数据）...")
X_filtered = X_scaled
vs_models = {}
for m_name in ["XGBoost", "RF", "SVR", "GPR"]:
    print(f"  训练 {m_name}...")
    if m_name == "XGBoost":
        m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    elif m_name == "RF":
        m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif m_name == "SVR":
        m = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    elif m_name == "GPR":
        m = _make_gpr()
    m.fit(X_filtered, y)
    vs_models[m_name] = m
print("全量模型训练完成！")

# ========== 5. 批量预测 ==========
tier_labels = {1: "高", 2: "中等", 3: "低", 4: "极低"}
tier_explanations = {
    1: "阴阳离子均在训练集出现（分别出现），历史LOIO-CV表现良好",
    2: "新阴离子TFA，有化学类似物（羧酸根家族：OAc, L-lactate, CH3CH2CO2）",
    3: "新阴离子SCN，化学类似物有限（仅与TFSI/TfO共享S原子）",
    4: "新阴离子DCN，无化学类似物，EtSO4前车之鉴（R^2=-82），预测仅供参考",
}

all_vs_results = {}
for T_cond, P_cond in [(303.15, 1.0), (313.15, 1.0)]:
    print(f"\n{'='*80}")
    print(f"=== 虚拟筛选结果: T={T_cond}K, P={P_cond}bar ===")
    print(f"{'='*80}")

    results = []
    for cat, ani, tier in all_pool:
        il_short = f"[{cation_short[cat]}][{anion_short[ani]}]"
        row = {"IL": il_short, "Tier": tier, "Cation_long": cat, "Anion_long": ani}
        cat_vec = cation_desc.loc[cat].values
        ani_vec = anion_desc.loc[ani].values
        feat = np.concatenate([cat_vec, ani_vec, [T_cond, P_cond]])
        feat = np.nan_to_num(feat, nan=0.0)
        feat = feat[~nan_cols]
        feat = feat[var_mask]
        feat_s = scaler.transform(feat.reshape(1, -1))
        preds = []
        for m_name in ["XGBoost", "RF", "SVR", "GPR"]:
            p = vs_models[m_name].predict(feat_s)[0]
            row[m_name] = round(p, 4)
            preds.append(p)
        row["Mean"] = round(np.mean(preds), 4)
        row["Std"] = round(np.std(preds), 4)
        row["Range"] = round(max(preds) - min(preds), 4)
        results.append(row)

    vs_df = pd.DataFrame(results)
    vs_df = vs_df.sort_values("Mean", ascending=False)

    # --- 输出：Top 20 ---
    print(f"\n{'─'*105}")
    print(f"  Top 20 (按四模型均值排序)")
    print(f"{'─'*105}")
    print(f"{'Rank':<5} {'IL':<22} {'Tier':<6} {'XGBoost':>8} {'RF':>8} {'SVR':>8} {'GPR':>8} {'Mean':>8} {'Std':>8} {'Range':>8}")
    print(f"{'─'*105}")
    for i, (_, r) in enumerate(vs_df.head(20).iterrows(), 1):
        print(f"{i:<5} {r['IL']:<22} {tier_labels[r['Tier']]:<6} {r['XGBoost']:>8.4f} {r['RF']:>8.4f} {r['SVR']:>8.4f} {r['GPR']:>8.4f} {r['Mean']:>8.4f} {r['Std']:>8.4f} {r['Range']:>8.4f}")

    # --- 分级统计 ---
    print(f"\n--- 分级统计 ---")
    for tier in [1, 2, 3, 4]:
        tier_df = vs_df[vs_df["Tier"] == tier]
        n = len(tier_df)
        if n == 0:
            continue
        top3 = tier_df.head(3)
        top3_str = ", ".join(f"{r['IL']}({r['Mean']:.4f})" for _, r in top3.iterrows())
        print(f"  Tier {tier} ({tier_labels[tier]}置信度): {n} ILs")
        print(f"    依据: {tier_explanations[tier]}")
        print(f"    Mean范围: [{tier_df['Mean'].min():.4f}, {tier_df['Mean'].max():.4f}], "
              f"Range中位数: {tier_df['Range'].median():.4f}, "
              f"Range>0.15: {(tier_df['Range']>0.15).sum()}/{n}")
        print(f"    Top 3: {top3_str}")

    # --- Tier 1 高共识推荐 ---
    tier1_df = vs_df[vs_df["Tier"] == 1].copy()
    tier1_consensus = tier1_df[tier1_df["Range"] < 0.10].sort_values("Mean", ascending=False)
    print(f"\n--- Tier 1 高共识推荐 (Range < 0.10, n={len(tier1_consensus)}) ---")
    if len(tier1_consensus) > 0:
        for i, (_, r) in enumerate(tier1_consensus.head(10).iterrows(), 1):
            print(f"  {i}. {r['IL']:<22} Mean={r['Mean']:.4f}  Range={r['Range']:.4f}")
    else:
        print("  (无满足条件的IL)")

    vs_df.to_csv(f'virtual_screening_full_{int(T_cond)}K.csv', index=False, encoding='utf-8-sig')
    print(f"\n已保存: virtual_screening_full_{int(T_cond)}K.csv ({len(vs_df)} ILs)")
    all_vs_results[T_cond] = vs_df

# ========== 6. 综合推荐 ==========
print(f"\n{'='*80}")
print("=== 虚拟筛选综合推荐 ===")
print(f"{'='*80}")
print("""
评价框架:
  无实验真值，用三个正交维度评估预测可信度:
  1. Tier (1-4): LOIO-CV校准的化学覆盖置信度
  2. Range (Max-Min): 四模型分歧，<0.05=一致, 0.05-0.15=可接受, >0.15=分歧大
  3. 跨温度趋势: 升温->溶解度下降 (物理合理)

推荐策略:
  第一优先: Tier 1 + Range < 0.10 的 Top 候选
  第二优先: Tier 1 + Range 0.10-0.15, 结合跨温度趋势判断
  不推荐: Tier 3-4 + Range > 0.15, 仅作方法演示
""")

# 跨温度一致性
print("--- 跨温度趋势 (303K -> 313K) ---")
if 303.15 in all_vs_results and 313.15 in all_vs_results:
    df303 = all_vs_results[303.15].set_index("IL")
    df313 = all_vs_results[313.15].set_index("IL")
    common_ils = df303.index.intersection(df313.index)
    deltas = []
    for il in common_ils:
        d = df313.loc[il, "Mean"] - df303.loc[il, "Mean"]
        deltas.append(d)
    n_increase = sum(1 for d in deltas if d > 0)
    n_decrease = sum(1 for d in deltas if d < 0)
    print(f"  {len(common_ils)} ILs: {n_decrease} 溶解度随升温下降(物理合理), "
          f"{n_increase} 上升(异常)")
    print(f"  Delta Mean范围: [{min(deltas):+.4f}, {max(deltas):+.4f}]")
    # 标记异常IL
    bad_mask = [d > 0.01 for d in deltas]
    if any(bad_mask):
        print(f"  异常IL (>0.01上升):")
        for il, d in zip(common_ils, deltas):
            if d > 0.01:
                print(f"    {il}: Delta={d:+.4f}")

# 终推荐
tier1_final = all_vs_results[303.15][all_vs_results[303.15]["Tier"] == 1]
tier1_best = tier1_final[tier1_final["Range"] < 0.10].sort_values("Mean", ascending=False)
print(f"\n=== 最终推荐 (Tier 1 + Range < 0.10) ===")
if len(tier1_best) >= 3:
    top3 = tier1_best.head(3)
    for i, (_, r) in enumerate(top3.iterrows(), 1):
        print(f"  {i}. {r['IL']}: Mean={r['Mean']:.4f}, Range={r['Range']:.4f} "
              f"(XGB={r['XGBoost']:.4f}, RF={r['RF']:.4f}, SVR={r['SVR']:.4f}, GPR={r['GPR']:.4f})")
else:
    print("  不足3个高共识候选，放宽至 Range < 0.15")
    tier1_relaxed = tier1_final[tier1_final["Range"] < 0.15].sort_values("Mean", ascending=False)
    for i, (_, r) in enumerate(tier1_relaxed.head(5).iterrows(), 1):
        print(f"  {i}. {r['IL']}: Mean={r['Mean']:.4f}, Range={r['Range']:.4f}")
print("=" * 80)

# ========== 主模型推荐（加权评分）==========
print("\n" + "="*60)
print("=== 主模型推荐（加权评分）===")
print("="*60)

# 逐IL汇总指标
per_il_stats = {}
for name, fold_df in loio_fold_results.items():
    valid = fold_df[fold_df['R2'].notna()]
    per_il_stats[name] = {
        "R2_gt_0_8_frac": (valid['R2'] > 0.8).sum() / len(fold_df),
        "IL_RMSE_median": fold_df['RMSE'].median(),
    }

# 评分权重: LOIO R² (35%) + MAE倒数 (25%) + R²>0.8占比 (25%) + IL_RMSE中位数倒数 (15%)
mae_inv = {r["model"]: 1.0 / r["MAE"] for r in loio_global_results}
rmse_med_inv = {name: 1.0 / per_il_stats[name]["IL_RMSE_median"] for name in per_il_stats}
mae_inv_max = max(mae_inv.values())
rmse_med_inv_max = max(rmse_med_inv.values())
r2_max = max(r["R2"] for r in loio_global_results)
frac_max = max(per_il_stats[name]["R2_gt_0_8_frac"] for name in per_il_stats)

scores = {}
for r in loio_global_results:
    name = r["model"]
    s = (0.35 * (r["R2"] / r2_max) +
         0.25 * (mae_inv[name] / mae_inv_max) +
         0.25 * (per_il_stats[name]["R2_gt_0_8_frac"] / frac_max) +
         0.15 * (rmse_med_inv[name] / rmse_med_inv_max))
    scores[name] = s

ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print("\n综合排序:")
for i, (name, s) in enumerate(ranking, 1):
    r = next(row for row in loio_global_results if row["model"] == name)
    print(f"  {i}. {name}: score={s:.4f}  (LOIO R²={r['R2']:.4f}, MAE={r['MAE']:.4f}, "
          f"R²>0.8={per_il_stats[name]['R2_gt_0_8_frac']:.1%}, "
          f"IL RMSE med={per_il_stats[name]['IL_RMSE_median']:.4f})")

print(f"\n推荐主模型: {ranking[0][0]}")
print("="*60)