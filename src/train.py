import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from evaluate import _make_gpr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    vs_models = {}
    X = np.load(MODELS_DIR / "X_raw.npy")
    y = np.load(MODELS_DIR / "y_raw.npy")
    print("加载数据并训练全量模型...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    np.save(MODELS_DIR / "X_scaled.npy", X_scaled)  #现在保存的是干净数据
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("特征矩阵和标签已保存。")
    #X_scaled已在构建阶段完成列过滤，直接使用
    X_filtered = X_scaled
    
    #训练最终模型 (复用已验证过的超参数)
    final_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    #训练四模型（全量数据）
    print("\n正在为虚拟筛选训练四模型（全量数据）...")
    X_filtered = X_scaled
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
        
    joblib.dump(vs_models,MODELS_DIR / "vs_models.pkl")
    final_model.fit(X_filtered, y)
    print("模型训练完成！")
    joblib.dump(final_model, MODELS_DIR / "final_model.pkl")
    print("模型已成功使用joblib保存！")

if __name__ == "__main__":
    main()
