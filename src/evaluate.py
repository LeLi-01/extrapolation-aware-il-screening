from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
# 虽然它们都在 sklearn.gaussian_process 这个“大柜子”里，但 Python 的导入机制不会因为导入了】
# GaussianProcessRegressor 就自动把所有核函数也带进来。
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
# 拿 随机划分函数，用来生成训练/测试集，评估模型在随机采样下的表现。
from sklearn.model_selection import train_test_split
# from preprocess import cation_smiles, anion_smiles


# =========================
# 0. 项目路径
# =========================
# 1. 加载数据（建议从保存的 il_ids.npy 读取，避免顺序风险）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

loio_global_results = []
loio_fold_results = {}
loio_predictions = {}

def _make_gpr():
    """GPR每折需重新实例化（fit后内核状态不可重用）
    RBF length_scale=10.0适配~200维特征空间（典型点间距~20）
    WhiteKernel noise=0.1（信噪比≈10），避免noise=1.0导致的过度平滑"""
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(0.1),
        alpha=1e-5, normalize_y=True, random_state=42,
        optimizer=None,
    )

random_models = {
    "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0),
    "RF": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.01),
    "GPR": _make_gpr(),
}



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

def main():
    print("开始评估...")
    # 创建目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "zhao_data.csv")
    X_raw = np.load(MODELS_DIR / "X_raw.npy")
    y = np.load(MODELS_DIR / "y_raw.npy")
    IL_ID = np.load(MODELS_DIR / "IL_ID.npy")


    # 验证顺序一致性（可选）
    print(f"X_raw shape: {X_raw.shape}, y shape: {y.shape}, IL_IDs length: {len(IL_ID)}")

    # 2. 处理 NaN / Inf
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. 确定所有 IL 种类
    unique_ILs = np.unique(IL_ID)
    print(f"总样本数: {len(y)}, 原始特征维度: {X_raw.shape[1]}, IL 种类: {len(unique_ILs)}")

    print("\n" + "=" * 50)
    print("=== 随机划分 80/20 多模型对比 ===")
    print("=" * 50)
    random_results_list = []
    for name, model in random_models.items():
        print(f"\n--- 随机划分基线: {name} ---")
        res = run_random_split_baseline(model, name, X_raw, y)
        random_results_list.append(res)
        print(f"  {name} 平均 R²: {res['R2_mean']:.4f} ± {res['R2_std']:.4f}")

    random_comparison = pd.DataFrame(random_results_list)
    print("\n" + random_comparison.to_string(index=False))
    random_comparison.to_csv(OUTPUTS_DIR / "random_split_model_comparison.csv", index=False)
    # 这是 Pandas DataFrame 自带的一个方法。它不直接打印，而是把表格里的数据转换成一段非常规整的纯文本字符串。
    print("已保存: random_split_model_comparison.csv")

    for name, model in loio_models.items():
        print(f"\n{'=' * 40}")
        print(f"开始执行 LOIO-CV: {name}")
        print(f"{'=' * 40}")
        global_m, fold_df, yt, yp = run_loio_cv(model, name, unique_ILs, IL_ID, X_raw, y)
        loio_global_results.append(global_m)
        loio_fold_results[name] = fold_df
        loio_predictions[name] = (yt, yp)
        # XGBoost 打印单折（论文表4需要），RF 仅汇总
        if name == "XGBoost":
            for _, row in fold_df.iterrows():
                # _：这是一个约定俗成的变量名，表示“我不关心这个值”。在这里代表行索引（0, 1, 2...），
                # 因为你不需要它，所以用 _ 丢弃了。
                print(
                    f"  {row['IL_ID']:<15} | R²: {row['R2']:.4f} | RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f} | n: {int(row['n_test'])}")
        print(f"\n=== LOIO-CV ({name}) 全局评估 ===")
        print(f"总测试样本数 : {global_m['n_total']}")
        print(f"全局 R²      : {global_m['R2']:.4f}")
        print(f"全局 RMSE    : {global_m['RMSE']:.4f}")
        print(f"全局 MAE     : {global_m['MAE']:.4f}")

    # 保存四种模型每折结果
    for name, fold_df in loio_fold_results.items():
        fold_df.to_csv(OUTPUTS_DIR / f"loio_cv_{name}_per_il.csv", index=False)
    print(" \n已保存四种模型逐 IL 结果: loio_cv_*_per_il.csv")

    # 保留 XGBoost 结果给后续分析
    results_df = loio_fold_results["XGBoost"]

    # --- 5月28日：多模型 LOIO-CV 对比汇总 ---
    print("\n" + "=" * 50)
    print("=== LOIO-CV 多模型对比汇总 ===")
    loio_comparison = pd.DataFrame(loio_global_results)
    print(loio_comparison.to_string(index=False))
    # loio_comparison.to_csv("loio_cv_model_comparison.csv", index=False)
    loio_comparison.to_csv(
        OUTPUTS_DIR / "loio_cv_model_comparison.csv",
        index=False
    )
    print("已保存: loio_cv_model_comparison.csv")
    # --- LOIO-CV vs 随机划分 综合对比表 ---
    print("\n" + "=" * 60)
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
    summary_df.to_csv(OUTPUTS_DIR / "loio_vs_random_comparison.csv", index=False)
    print("\n已保存: loio_vs_random_comparison.csv")

    # --- 5月28日：多维度对比表（含逐IL RMSE分布、R²>0.5占比等）---
    print("\n" + "=" * 80)
    print("=== 多维度综合对比 ===")
    print("=" * 80)
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
            "R²>0.5占比": f"{(valid['R2'] > 0.5).sum()}/{n_total} ({(valid['R2'] > 0.5).sum() / n_total:.1%})",
            "R²>0.8占比": f"{(valid['R2'] > 0.8).sum()}/{n_total} ({(valid['R2'] > 0.8).sum() / n_total:.1%})",
            "R²<0占比": f"{(valid['R2'] < 0).sum()}/{n_total} ({(valid['R2'] < 0).sum() / n_total:.1%})",
        })
    multi_df = pd.DataFrame(multi_rows)
    print(multi_df.to_string(index=False))
    multi_df.to_csv(OUTPUTS_DIR / "multimetric_comparison.csv", index=False)
    print("\n已保存: multimetric_comparison.csv")


if __name__ == "__main__":
    main()