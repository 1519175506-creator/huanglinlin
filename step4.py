#!/usr/bin/env python3
"""
FINA2003 模块（三）风险初探
将预测得到的「海关大类」作为新特征，映射风险档位；叠加《保险除外标的表》关键词兜底。

两种用法：
1）被 cargo_classification_analysis.py 调用：run_risk_module(cargo, excl, out_dir)
2）独立运行：需先有 output/cargo_with_predictions.csv（先跑完整分类主脚本生成）

依赖：pandas、numpy、matplotlib；数据路径默认与项目根目录相对。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "output"
DEFAULT_OUT.mkdir(exist_ok=True)
_mpl = DEFAULT_OUT / ".mplconfig"
_mpl.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PATH_EXCLUDE_DEFAULT = ROOT / "保险除外标的表.xlsx"
PATH_PRED_DEFAULT = DEFAULT_OUT / "cargo_with_predictions.csv"


def load_exclusion_keywords(excl: pd.DataFrame) -> list[str]:
    col = excl.columns[0]
    keys: list[str] = []
    for v in excl[col].dropna():
        s = str(v)
        parts = re.split(r"[；，、（）0-9]", s)
        for p in parts:
            p = p.strip()
            if len(p) >= 2:
                keys.append(p)
    keys = sorted(set(keys), key=len, reverse=True)
    return keys


RISK_MAJOR_CATEGORY_HEURISTIC: dict[str, str] = {
    "武器、弹药及其零件、附件": "高",
    "活动物;动物产品": "高",
    "艺术品、收藏品及古物": "高",
    "化学工业及其相关工业的产品": "中高",
    "食品；饮料、酒及醋；烟草、烟草及烟草代用品的制品": "中",
    "植物产品": "中",
    "矿产品": "中低",
    "纺织原料及纺织制品": "低",
    "贱金属及其制品": "低",
    "塑料及其制品；橡胶及其制品": "低",
    "机器、机械器具、电气设备及其零件；录音机及放声机、电视图像、声音的录制和重放设备及其零件、附件": "中",
    "车辆、航空器、船舶及有关运输设备": "高",
    "杂项制品": "中",
    "石料、石膏、水泥、石棉、云母及类似材料的制品；陶瓷产品；玻璃及其制品": "中高",
    "光学、照相、电影、计量、检验、疗或外科用仪器及设备、精密仪器及设备；钟表；乐器；上述物品的零件、附件": "中高",
    "木及木制品；木炭；软木及软木制品；稻草、秸秆、针茅或其他编结材料制品；篮筐及柳条编结品": "低",
    "木浆及其他纤维状纤维素浆；回收（废碎）纸及纸板": "低",
    "生皮、皮革、毛皮及其制品；鞍具及挽具；旅游用品、手提包及类似容器；动物肠线": "低",
    "天然或养殖珍珠、宝石或半宝石、贵金属、包贵金属及其制品；仿首饰；硬币": "高",
    "鞋、帽、伞、杖、鞭及其零件；已加工的羽毛及其制品；人造花；人发制品": "低",
    "动、植物油、脂及其分解产品；精致的食用油脂；动、植物蜡": "中",
    "特殊交易品及未分类商品": "中",
}


def risk_from_prediction(predicted_major: str, raw_desc: str, excl_keywords: list[str]) -> str:
    base = RISK_MAJOR_CATEGORY_HEURISTIC.get(predicted_major, "中")
    t = str(raw_desc)
    for kw in excl_keywords:
        if kw and kw in t:
            return "高（除外条款命中）"
    return base


_RISK_SCORE_MAP = {
    "低": 1,
    "中低": 2,
    "中": 3,
    "中高": 4,
    "高": 5,
    "高（除外条款命中）": 5,
}


def risk_tier_to_binary_high(tier: str) -> int:
    if tier.startswith("高"):
        return 1
    if tier == "中高":
        return 1
    return 0


def risk_tier_to_score(tier: str) -> int:
    return int(_RISK_SCORE_MAP.get(tier, 3))


def plot_risk_bar(counts: pd.Series, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.iloc[::-1].plot(kind="barh", ax=ax, color="#c0504d")
    ax.set_title(title)
    ax.set_xlabel("条数")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_risk_by_predicted_major(cargo: pd.DataFrame) -> pd.DataFrame:
    g = cargo.groupby("pred_major", dropna=False).agg(
        n=("risk_high_flag", "count"),
        high_share=("risk_high_flag", "mean"),
        mean_risk_score=("risk_score", "mean"),
        mean_cosine=("l2_cosine_sim", "mean"),
        lowconf_share=("classifier_lowconf", "mean"),
    )
    return g.sort_values("n", ascending=False)


def run_risk_module(cargo: pd.DataFrame, excl: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    输入 cargo 须含列：desc_raw, pred_major, pred_hs6（可选展示）, match_layer, l2_cosine_sim。
    输出：写入 risk_*.csv / 图，并在 cargo 上增加 risk_level, risk_high_flag, risk_score, classifier_lowconf。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    excl_kw = load_exclusion_keywords(excl)

    cargo = cargo.copy()
    cargo["risk_level"] = [
        risk_from_prediction(pm, raw, excl_kw)
        for pm, raw in zip(cargo["pred_major"], cargo["desc_raw"])
    ]
    cargo["risk_high_flag"] = cargo["risk_level"].map(risk_tier_to_binary_high).astype(np.int8)
    cargo["risk_score"] = cargo["risk_level"].map(risk_tier_to_score).astype(np.int8)
    cargo["classifier_lowconf"] = (cargo["match_layer"] == "L2_lowconf").astype(np.int8)

    risk_counts = cargo["risk_level"].value_counts()
    risk_counts.to_csv(out_dir / "risk_level_counts.csv", encoding="utf-8-sig")
    plot_risk_bar(
        risk_counts,
        out_dir / "risk_level_bar.png",
        title="基于预测大类+除外规则的风险档位条数分布",
    )

    high_share = float(cargo["risk_high_flag"].mean())
    print("【模块三 — 新特征定义】")
    print(
        "- feat_pred_major：由预测 HS6 反查海关「大类名称」（分类流水线输出）。\n"
        "- feat_risk_level：大类 → 启发式风险档；文本命中除外标的则覆盖为「高（除外条款命中）」。\n"
        "- feat_risk_high_flag：保守二元——「高」「中高」「高（除外…）」记 1。\n"
        "- feat_risk_score：有序分值 1–5。\n"
        "- feat_classifier_lowconf：match_layer==L2_lowconf 时记 1。\n"
    )
    print(
        f"【模块三 — 二元高风险倾向占比】risk_high_flag=1 占比: {high_share:.2%} "
        f"（{int(cargo['risk_high_flag'].sum())} / {len(cargo)}）\n"
    )

    cargo["pred_major"].value_counts().head(15).to_csv(
        out_dir / "cargo_predicted_major_top15.csv",
        encoding="utf-8-sig",
    )

    risk_by_major = summarize_risk_by_predicted_major(cargo)
    risk_by_major.to_csv(out_dir / "risk_summary_by_pred_major.csv", encoding="utf-8-sig")

    ct_layer_high = pd.crosstab(cargo["match_layer"], cargo["risk_high_flag"], margins=True)
    ct_layer_high.to_csv(out_dir / "risk_high_flag_crosstab_layer.csv", encoding="utf-8-sig")

    cols_feat = [
        "desc_raw",
        "pred_hs6",
        "pred_major",
        "match_layer",
        "l2_cosine_sim",
        "risk_level",
        "risk_high_flag",
        "risk_score",
        "classifier_lowconf",
    ]
    present = [c for c in cols_feat if c in cargo.columns]
    cargo[present].head(800).to_csv(
        out_dir / "cargo_risk_features_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_cols = [c for c in cols_feat if c in cargo.columns]
    cargo[full_cols].to_csv(out_dir / "cargo_with_risk_features.csv", index=False, encoding="utf-8-sig")

    print("【模块三 — 业务含义（可写入报告）】")
    print(
        "- 在缺少历史理赔标注的前提下，将「自动归类结果」视为结构化风险特征。\n"
        "- 二元特征用于筛查队列优先级；分类错误会传导至风险误判，须与低置信标记联动。\n"
    )
    print("【风险档位计数】")
    print(risk_counts.to_string())
    print()
    print("【预测大类 × 高风险倾向（Top10 样本量）】")
    print(risk_by_major.head(10).to_string())
    print()
    print("【match_layer × risk_high_flag】")
    print(ct_layer_high.to_string())
    print()
    print(f"模块三输出已写入: {out_dir}")
    return cargo


def main():
    print("=== FINA2003 模块（三）风险初探（独立运行）===\n")
    pred_path = PATH_PRED_DEFAULT
    if not pred_path.is_file():
        print(
            f"未找到分类结果文件: {pred_path}\n"
            "请先运行: python3 cargo_classification_analysis.py\n"
            "（主脚本会在 output/ 生成 cargo_with_predictions.csv）"
        )
        return

    cargo = pd.read_csv(pred_path, encoding="utf-8-sig")
    required = {"desc_raw", "pred_major", "match_layer", "l2_cosine_sim"}
    missing = required - set(cargo.columns)
    if missing:
        print(f"CSV 缺少列: {missing}")
        return

    if "pred_hs6" not in cargo.columns:
        cargo["pred_hs6"] = ""

    excl_path = PATH_EXCLUDE_DEFAULT
    if not excl_path.is_file():
        print(f"未找到除外标的表: {excl_path}")
        return
    excl = pd.read_excel(excl_path)

    run_risk_module(cargo, excl, DEFAULT_OUT)


if __name__ == "__main__":
    main()
