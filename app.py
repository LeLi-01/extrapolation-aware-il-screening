"""
Streamlit V1.0 界面：离子液体 H2S 溶解度预测、批量筛选与外推可靠性评估。

复用 src/inference.py 的推理接口，不修改模型权重、不重新训练。

注意：本环境 pyarrow 与 protobuf/absl 的 C++ 库存在 ABI 冲突，导致
st.dataframe/st.table/st.bar_chart 等组件无法使用，因此表格改用 pandas 的
to_html + st.markdown 渲染，图表改用 matplotlib + st.pyplot，均不依赖 pyarrow。

启动：在项目根目录执行
    streamlit run app.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.preprocess import anion_smiles, cation_smiles
from src.inference import (
    MODEL_NAMES,
    predict_il,
    predict_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parent
LOIO_COMPARISON = PROJECT_ROOT / "outputs" / "loio_cv_model_comparison.csv"

st.set_page_config(page_title="离子液体智能筛选与外推可靠性评估系统", layout="wide")

# 表格样式
st.markdown(
    """
    <style>
    table.dataframe { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }
    table.dataframe th, table.dataframe td {
        border: 1px solid #e0e0e0; padding: 6px 12px; text-align: left;
        font-size: 0.9rem;
    }
    table.dataframe th { background: #f0f2f6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 公共工具
# ============================================================
def render_table(df: pd.DataFrame):
    """用 pandas.to_html 渲染表格（不依赖 pyarrow）。"""
    st.markdown(df.to_html(index=False, classes="dataframe"), unsafe_allow_html=True)


def render_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """matplotlib 柱状图（不依赖 pyarrow）。"""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    """matplotlib 散点图（不依赖 pyarrow）。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def ion_input(prefix: str):
    """阳离子、阴离子输入：下拉选择（全名）或手输 SMILES。返回 (cation, anion)。"""
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("阳离子输入方式", ["下拉选择", "输入 SMILES"],
                        horizontal=True, key=f"{prefix}_cat_mode")
        if mode == "下拉选择":
            cation = st.selectbox("阳离子", list(cation_smiles.keys()),
                                  key=f"{prefix}_cat_select")
        else:
            cation = st.text_input("阳离子 SMILES", value="CCCC[n+]1ccn(C)c1",
                                   key=f"{prefix}_cat_smi")
    with col2:
        mode = st.radio("阴离子输入方式", ["下拉选择", "输入 SMILES"],
                        horizontal=True, key=f"{prefix}_ani_mode")
        if mode == "下拉选择":
            anion = st.selectbox("阴离子", list(anion_smiles.keys()),
                                 key=f"{prefix}_ani_select")
        else:
            anion = st.text_input("阴离子 SMILES",
                                  value="O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
                                  key=f"{prefix}_ani_smi")
    return cation, anion


def condition_input(prefix: str):
    """温度、压力输入。返回 (T, P)。"""
    col1, col2 = st.columns(2)
    T = col1.number_input("温度 T (K)", min_value=200.0, max_value=600.0,
                          value=303.15, step=1.0, key=f"{prefix}_T")
    P = col2.number_input("压力 P (bar)", min_value=0.001, max_value=200.0,
                          value=1.0, step=0.1, key=f"{prefix}_P")
    return T, P


def tier_badge(result: dict):
    """可靠分级徽章。"""
    tier = result["tier"]
    if tier == "内插（组合已知）":
        st.success(f"可靠分级：{tier}")
    elif tier == "Tier 1":
        st.warning(f"可靠分级：{tier}")
    else:
        st.error(f"可靠分级：{tier}")
    st.caption(result["tier_note"])


def result_table(result: dict) -> pd.DataFrame:
    """四模型预测结果表。"""
    rows = []
    for m in MODEL_NAMES:
        rows.append({"模型": m, "预测值（摩尔分数）": round(result[m], 4)})
    rows.append({"模型": "四模型均值 Mean", "预测值（摩尔分数）": round(result["mean"], 4)})
    rows.append({"模型": "模型间标准差 Std", "预测值（摩尔分数）": round(result["std"], 4)})
    rows.append({"模型": "模型间极差 Range", "预测值（摩尔分数）": round(result["range"], 4)})
    return pd.DataFrame(rows)


@st.cache_data
def load_loio_reference() -> pd.DataFrame | None:
    if LOIO_COMPARISON.exists():
        return pd.read_csv(LOIO_COMPARISON)
    return None


# ============================================================
# 溶解度预测：H2S 溶解度预测
# ============================================================
def page_predict():
    st.header("硫化氢溶解度预测")
    st.caption("输入离子液体结构（阳离子/阴离子）与工况（温度、压力），"
               "预测硫化氢溶解度（摩尔分数）。本模型基于 H₂S–IL 数据构建，"
               "作为筛选硫醇前置吸收离子液体的代理指标，不直接预测硫醇溶解度。")

    cation, anion = ion_input("p1")
    T, P = condition_input("p1")

    if st.button("预测", type="primary"):
        try:
            with st.spinner("推理中..."):
                result = predict_il(cation, anion, T, P)
        except ValueError as e:
            st.error(f"输入有误：{e}")
            return

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("主要预测值（四模型均值）",
                      f"{result['mean']:.4f}",
                      help="H2S 溶解度，单位：摩尔分数")
            st.caption("单位：摩尔分数 (mole fraction)")
        with col2:
            render_table(result_table(result))

        tier_badge(result)

        if not result["T_in_training_range"] or not result["P_in_training_range"]:
            st.warning("当前工况超出训练数据覆盖范围（T: 293.15–403.15 K，"
                       "P: 0.001–96.3 bar），结果属外推，可靠性降低。")


# ============================================================
# 候选材料筛选：候选离子液体筛选（CSV 批量预测）
# ============================================================
def _errors_table(errors: list[dict]) -> pd.DataFrame:
    """把 predict_batch 的校验失败记录转成展示用 DataFrame。"""
    return pd.DataFrame([{
        "行号": e["row"],
        "Cation": e["Cation"],
        "Anion": e["Anion"],
        "T": e["T"],
        "P": e["P"],
        "错误原因": "；".join(e["errors"]),
    } for e in errors])


def _download_errors(errors: list[dict]):
    if not errors:
        return
    st.download_button(
        "导出校验失败记录 CSV",
        data=_errors_table(errors).to_csv(index=False).encode("utf-8-sig"),
        file_name="h2s_screening_errors.csv",
        mime="text/csv",
    )


def page_batch():
    st.header("候选离子液体筛选（CSV 批量预测）")
    st.caption("上传 CSV 文件批量预测，逐行校验输入并按外推风险排序。"
               "CSV 需包含列：Cation、Anion、T、P（温度单位 K，压力单位 bar；"
               "也兼容列名 T(K)/P(bar)）。Cation/Anion 列可填全名、短名或 SMILES。")

    with st.expander("查看 CSV 列格式说明"):
        st.code(
            "Cation,Anion,T,P\n"
            "C4MIm or BMIM,TFSI or NTf2,303.15,1.0\n"
            "EMIM or C2MIm,BF4,303.15,1.0\n"
            "CCCC[n+]1ccn(C)c1,O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F,313.15,2.0\n",
            language="text",
        )

    uploaded = st.file_uploader("上传 CSV", type=["csv"])
    if uploaded is None:
        st.info("请上传 CSV 文件以开始批量预测。")
        return

    df = pd.read_csv(uploaded)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "cation":
            rename[c] = "Cation"
        elif lc == "anion":
            rename[c] = "Anion"
        elif lc in ("t", "t(k)", "temperature"):
            rename[c] = "T"
        elif lc in ("p", "p(bar)", "pressure"):
            rename[c] = "P"
    df = df.rename(columns=rename)

    required = {"Cation", "Anion", "T", "P"}
    if not required.issubset(df.columns):
        st.error(f"CSV 缺少必需列，需包含 {sorted(required)}。当前列：{list(df.columns)}")
        return

    with st.spinner(f"批量预测 {len(df)} 条..."):
        records = [
            {"Cation": row["Cation"], "Anion": row["Anion"],
             "T": row["T"], "P": row["P"]}
            for _, row in df.iterrows()
        ]
        results, errors = predict_batch(records)

    if not results:
        st.error("没有成功预测任何一行。")
        if errors:
            st.subheader("校验失败明细")
            render_table(_errors_table(errors))
            _download_errors(errors)
        return

    out = pd.DataFrame([{
        "Cation": r["cation"],
        "Anion": r["anion"],
        "T(K)": r["T"],
        "P(bar)": r["P"],
        **{m: round(r[m], 4) for m in MODEL_NAMES},
        "Mean": round(r["mean"], 4),
        "Range": round(r["range"], 4),
        "Tier": r["tier"],
        "T_in_range": "是" if r["T_in_training_range"] else "否",
        "P_in_range": "是" if r["P_in_training_range"] else "否",
        "Risk": r["risk_status"],
        "异常模型": "、".join(r["out_of_range_models"]) if r["out_of_range_models"] else "无",
    } for r in results])

    # 可靠性优先排序：正常 -> 工况外推 -> 物理范围异常；同档内按均值降序。
    risk_order = {"正常": 0, "工况外推": 1, "物理范围异常": 2}
    out["_risk_rank"] = out["Risk"].map(lambda x: risk_order.get(x, 3))
    out = (
        out.sort_values(["_risk_rank", "Mean"], ascending=[True, False], kind="stable")
        .drop(columns="_risk_rank")
        .reset_index(drop=True)
    )

    n_physical = int((out["Risk"] == "物理范围异常").sum())
    n_extrap = int((out["Risk"] == "工况外推").sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("成功预测", len(results))
    col2.metric("校验失败", len(errors))
    col3.metric("物理范围异常", n_physical)

    st.subheader("预测结果（按可靠性排序：正常 → 工况外推 → 物理范围异常）")
    render_table(out)

    if n_physical:
        st.error(f"{n_physical} 条记录的模型预测值超出摩尔分数 [0,1] 物理范围，"
                 "已排至末尾并标记，请勿作为筛选依据。")
    if n_extrap:
        st.warning(f"{n_extrap} 条记录的温度/压力超出训练覆盖范围（工况外推），"
                   "结果属外推，可靠性降低。")

    if len(out) > 1:
        viz = out.sort_values("Mean", ascending=False)
        st.subheader("候选离子液体预测结果可视化")
        render_bar_chart(viz.head(20), "Cation", "Mean",
                         "Candidate Ionic Liquid Predictions (Four-Model Mean)")

        st.subheader("预测值 vs 模型间分歧")
        render_scatter(out, "Mean", "Range")

    st.download_button(
        "导出预测结果 CSV",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name="h2s_screening_results.csv",
        mime="text/csv",
    )

    if errors:
        st.warning(f"{len(errors)} 行校验失败，已跳过并单独导出。")
        with st.expander("查看校验失败明细"):
            render_table(_errors_table(errors))
        _download_errors(errors)


# ============================================================
# 页面三：外推可靠性评估
# ============================================================
def page_reliability():
    st.header("外推可靠性评估")
    st.caption("对单个候选离子液体，从化学空间覆盖（Tier）、多模型一致性（Range）、"
               "工况适用范围三个维度评估外推可靠性。")

    cation, anion = ion_input("p3")
    T, P = condition_input("p3")

    if st.button("评估可靠性", type="primary"):
        try:
            result = predict_il(cation, anion, T, P)
        except ValueError as e:
            st.error(f"输入有误：{e}")
            return

        st.subheader("当前候选的可靠性评估")
        tier_badge(result)

        col1, col2, col3 = st.columns(3)
        col1.metric("模型间极差 Range", f"{result['range']:.4f}",
                    help="四模型预测值的 max−min")
        col2.metric("温度在训练范围内", "是" if result["T_in_training_range"] else "否")
        col3.metric("压力在训练范围内", "是" if result["P_in_training_range"] else "否")

        range_note = (
            "模型间高度一致" if result["range"] < 0.05
            else "模型间可接受" if result["range"] < 0.15
            else "模型间分歧较大，结果需谨慎"
        )
        st.caption(f"Range 解读：{range_note}（<0.05 一致 / 0.05–0.15 可接受 / >0.15 分歧大）")

        st.subheader("历史 LOIO-CV 指标")
        loio = load_loio_reference()
        if loio is not None:
            disp = loio.copy()
            disp["R2"] = disp["R2"].round(3)
            disp["RMSE"] = disp["RMSE"].round(4)
            disp["MAE"] = disp["MAE"].round(4)
            render_table(disp)
            st.caption("LOIO-CV（留一离子液体交叉验证）衡量模型对未见离子液体的历史泛化表现。"
                       "RMSE/MAE 单位为摩尔分数。")
        else:
            st.info("未找到 LOIO-CV 结果文件（outputs/loio_cv_model_comparison.csv）。")

        st.warning("重要说明：历史 LOIO-CV 指标是模型在全体训练数据上的统计表现，"
                   "不代表当前单个候选的预测误差，仅供判断外推风险的参考。")


# ============================================================
# 入口
# ============================================================
def main():
    st.sidebar.title("离子液体智能筛选与外推可靠性评估系统")
    st.sidebar.caption("基于机器学习的硫化氢溶解度预测与硫醇前置吸收剂辅助筛选")
    page = st.sidebar.radio(
        "导航",
        ["溶解度预测", "候选离子液体批量筛选", "外推可靠性评估"],
    )

    if page == "溶解度预测":
        page_predict()
    elif page == "候选离子液体批量筛选":
        page_batch()
    else:
        page_reliability()


if __name__ == "__main__":
    main()
