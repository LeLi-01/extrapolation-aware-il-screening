import numpy as np
import joblib
import pandas as pd

from preprocess import cation_smiles, anion_smiles
from preprocess import compute_descriptors

from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SCALER_PKL =MODELS_DIR / "scaler.pkl"
VAR_MASK_NPY = MODELS_DIR / "var_mask.npy"
NAN_COLS_MASK = MODELS_DIR / "nan_cols_mask.npy"
VS_MODELS_PKL = MODELS_DIR / "vs_models.pkl"


# ========== 构建筛选池 ==========
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

def main():
    # 创建目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


    # =========================================================================
    # 虚拟筛选：全组合扫描，四模型共识预测 + 分级置信度
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== 虚拟筛选：全组合扫描 ===")
    print("=" * 80)

    # ========== 加载预处理对象 ==========
    scaler = joblib.load(SCALER_PKL)
    var_mask = np.load(VAR_MASK_NPY)
    nan_cols = np.load(NAN_COLS_MASK)
    df = pd.read_csv(DATA_DIR / "zhao_data.csv")
    vs_models = joblib.load(VS_MODELS_PKL)

    # 加载 LOIO 全局评价结果
    loio_global_results = pd.read_csv(
        OUTPUTS_DIR / "loio_cv_model_comparison.csv"
    ).to_dict(orient="records")

    # 加载四个模型逐 IL 的评价结果
    loio_fold_results = {}

    for name in ["XGBoost", "RF", "SVR", "GPR"]:
        loio_fold_results[name] = pd.read_csv(
            OUTPUTS_DIR / f"loio_cv_{name}_per_il.csv"
        )

    train_cations = list(cation_smiles.keys())   # 7
    train_anions = list(anion_smiles.keys())      # 11
    measured_pairs = set()

    for _, row in df[['Cation', 'Anion']].drop_duplicates().iterrows():
        #整行去除
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

    # ==========  计算描述符 ==========
    cation_desc = compute_descriptors(all_cation_smiles)
    anion_desc = compute_descriptors(all_anion_smiles)

    # ========== 批量预测 ==========
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

        vs_df.to_csv(OUTPUTS_DIR / f'virtual_screening_full_{int(T_cond)}K.csv', index=False, encoding='utf-8-sig')
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

if __name__ == "__main__":
    main()