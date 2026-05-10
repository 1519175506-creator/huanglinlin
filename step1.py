#!/usr/bin/env python3
"""
Step 1 — 模块（二）2.1 数据探索（整合版）

合并自：
- Fintech0510.py：长度分布、HS 提取与大类关联、噪声统计、数据质量摘要、词云等
- 0510.py：中英文本分离后再分词，修正「全库拼接 + 中文词被大写破坏」问题

输出默认写入项目 `output/`，文件名带 `eda_` 前缀。
"""
from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
PATH_CARGO = PROJECT_ROOT / "货物描述信息2025.xlsx"
PATH_SHCODE = PROJECT_ROOT / "海关shcode表.xlsx"
PATH_EXCLUDE = PROJECT_ROOT / "保险除外标的表.xlsx"
OUT = PROJECT_ROOT / "output"

_mpl = OUT / ".mplconfig"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_cargo() -> pd.DataFrame:
    if not PATH_CARGO.is_file():
        raise FileNotFoundError(f"未找到货描文件: {PATH_CARGO}")
    try:
        return pd.read_excel(PATH_CARGO, sheet_name="数据")
    except ValueError:
        return pd.read_excel(PATH_CARGO)


def _normalize_desc_column(df: pd.DataFrame) -> pd.DataFrame:
    for cand in ("货物描述itemDescription", "货物描述", "itemDescription"):
        if cand in df.columns:
            return df.rename(columns={cand: "itemDescription"})
    return df.rename(columns={df.columns[0]: "itemDescription"})


def separate_cn_en(text) -> tuple[str, list[str]]:
    """分离中文连续片段与英文词（0510 修正版逻辑）。"""
    text = str(text).strip()
    cn_chars = re.findall(r"[\u4e00-\u9fff]+", text)
    cn_text = " ".join(cn_chars)
    en_text = re.sub(r"[\u4e00-\u9fff]", " ", text)
    en_text = re.sub(r"[^\w\s]", " ", en_text)
    en_words = [w for w in en_text.split() if w.isalpha() and len(w) > 1]
    return cn_text, en_words


def tokenize_cn(text: str, stopwords_cn: set[str]) -> list[str]:
    words: list[str] = []
    for word in jieba.cut(text):
        word = word.strip()
        if len(word) > 1 and word not in stopwords_cn:
            words.append(word)
    return words


def extract_hs_code(text) -> str | None:
    """从描述中提取 HS（前 6 位），与 Fintech0510 规则一致。"""
    patterns = [
        r"HS\s*CODE:?\s*(\d{4,10})",
        r"HS\s*CODE\s*NO:?\s*(\d{4,10})",
        r"HS:?\s*(\d{4,10})",
        r"HS\s*(\d{4,10})",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text).upper())
        if match:
            code = match.group(1)
            if len(code) >= 6:
                return code[:6]
            if len(code) >= 4:
                return code.ljust(6, "0")
    return None


def _shcode_hs6_str_series(sh: pd.DataFrame) -> pd.Series:
    def cell_to_hs6(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            if isinstance(v, (int, np.integer)):
                d = str(int(v))
            else:
                s = str(v).strip()
                if not s or s.lower() == "nan":
                    return None
                if re.fullmatch(r"-?\d+(\.\d+)?", s):
                    d = re.sub(r"\D", "", str(int(float(s))))
                else:
                    d = re.sub(r"\D", "", s)
        except (ValueError, OverflowError):
            d = re.sub(r"\D", "", str(v))
        if len(d) >= 10:
            d = d[:10]
        if len(d) >= 6:
            return d[:6]
        return None

    return sh["HSCODE"].map(cell_to_hs6)


def detect_language(text) -> str:
    text = str(text)
    cn_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_words = len(re.findall(r"[a-zA-Z]+", text))
    if cn_chars > 0 and en_words > 0:
        return "中英混合"
    if cn_chars > 0:
        return "纯中文"
    if en_words > 0:
        return "纯英文"
    return "其他"


def _pick_wordcloud_font() -> str | None:
    for p in (
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(p).is_file():
            return p
    return None


def main():
    print("=" * 60)
    print("Step 1 — 数据探索（Fintech0510 + 0510 整合）")
    print("=" * 60)

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 数据加载 ----------
    print("\n1. 数据加载")
    df_cargo = _normalize_desc_column(_load_cargo())
    print(f"货物描述数据集形状: {df_cargo.shape}")
    print(f"列名: {df_cargo.columns.tolist()}")

    if not PATH_SHCODE.is_file():
        print(f"警告: 未找到海关表 {PATH_SHCODE}，跳过 HS 大类关联图")
        df_shcode = None
    else:
        df_shcode = pd.read_excel(PATH_SHCODE)
        print(f"海关 Shcode 表形状: {df_shcode.shape}")

    if PATH_EXCLUDE.is_file():
        df_exclude = pd.read_excel(PATH_EXCLUDE)
        print(f"除外标的表形状: {df_exclude.shape}")
    else:
        print(f"提示: 未找到 {PATH_EXCLUDE}（仅打印形状用途时可忽略）")

    # ---------- 行业词典 + 停用词（两脚本合并） ----------
    industry_words = list(
        dict.fromkeys(
            [
                "集装箱",
                "散装",
                "冷冻",
                "冷藏",
                "零部件",
                "配件",
                "原料",
                "半成品",
                "电动滑板车",
                "太阳能",
                "不锈钢",
                "铝合金",
                "碳纤维",
                "聚氨酯",
                "锂离子",
                "阀门",
                "石油钻机",
                "注塑",
                "冲压",
                "铸造",
                "化工",
                "纺织品",
                "服装",
                "电子产品",
                "机械设备",
                "汽车零配件",
                "玻璃",
                "陶瓷",
                "石材",
                "艺术品",
                "收藏品",
                "精密仪器",
                "烟花爆竹",
                "活动物",
                "鲜活",
                "易腐",
                "冷冻食品",
                "水彩颜料",
                "画笔",
                "绘画本",
                "健身器材",
                "圣诞树",
            ]
        )
    )
    for word in industry_words:
        jieba.add_word(word)

    stopwords_cn = set(
        "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 "
        "没有 看 好 自己 这 他 她 它 们 那 些 及 与 或 等 为 以".split()
    )
    stopwords_en = set(
        [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "including",
            "such",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "and",
            "or",
            "but",
            "not",
            "no",
            "if",
            "then",
            "than",
            "too",
            "very",
            "HS",
            "CODE",
            "NO",
            "CIF",
            "FOB",
            "AS",
            "PER",
            "INVOICE",
            "BL",
        ]
    )

    # ---------- 2. 基础统计 ----------
    print("\n2. 基础统计")
    print(df_cargo.isnull().sum())
    print(f"\n总样本数: {len(df_cargo)}")
    print(f"唯一描述数: {df_cargo['itemDescription'].nunique()}")
    print(f"重复描述数: {df_cargo['itemDescription'].duplicated().sum()}")

    # ---------- 3. 描述长度分析（Fintech0510） ----------
    print("\n3. 描述长度分析")
    df_cargo["char_length"] = df_cargo["itemDescription"].astype(str).str.len()
    df_cargo["word_count_cn"] = df_cargo["itemDescription"].astype(str).apply(
        lambda x: len([c for c in x if "\u4e00" <= c <= "\u9fff"])
    )
    df_cargo["word_count_en"] = df_cargo["itemDescription"].astype(str).apply(
        lambda x: len(re.findall(r"\b[a-zA-Z]+\b", x))
    )
    print(df_cargo["char_length"].describe())
    print(df_cargo["word_count_cn"].describe())
    print(df_cargo["word_count_en"].describe())

    df_cargo["language_type"] = df_cargo["itemDescription"].apply(detect_language)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(df_cargo["char_length"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    med = df_cargo["char_length"].median()
    axes[0, 0].axvline(med, color="red", linestyle="--", label=f"中位数: {med:.0f}")
    axes[0, 0].set_title("货品描述字符长度分布", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("字符长度")
    axes[0, 0].legend()

    length_bins = [0, 20, 50, 100, 200, 500, float("inf")]
    length_labels = ["≤20", "21-50", "51-100", "101-200", "201-500", ">500"]
    df_cargo["length_group"] = pd.cut(df_cargo["char_length"], bins=length_bins, labels=length_labels)
    length_dist = df_cargo["length_group"].value_counts().sort_index()
    axes[0, 1].bar(length_dist.index.astype(str), length_dist.values, color="coral", edgecolor="white", alpha=0.8)
    axes[0, 1].set_title("描述长度分组统计", fontsize=14, fontweight="bold")
    for i, v in enumerate(length_dist.values):
        axes[0, 1].text(i, v + 10, str(v), ha="center", fontsize=10)

    sample_idx = np.random.default_rng(42).choice(
        len(df_cargo), size=min(2000, len(df_cargo)), replace=False
    )
    axes[1, 0].scatter(
        df_cargo.iloc[sample_idx]["word_count_cn"],
        df_cargo.iloc[sample_idx]["word_count_en"],
        alpha=0.5,
        s=10,
        c="teal",
    )
    axes[1, 0].set_title("中文字数 vs 英文词数（采样）", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("中文字数")
    axes[1, 0].set_ylabel("英文词数")

    lang_dist = df_cargo["language_type"].value_counts()
    axes[1, 1].pie(
        lang_dist.values,
        labels=lang_dist.index,
        autopct="%1.1f%%",
        colors=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"],
        startangle=90,
    )
    axes[1, 1].set_title("语言类型分布", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "eda_描述长度分析.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {OUT / 'eda_描述长度分析.png'}")

    # ---------- 4. 词频（0510：中英分离后再统计） ----------
    print("\n4. 词频分析（中英分离 + jieba 中文）")
    all_cn_words: list[str] = []
    all_en_words: list[str] = []

    for desc in df_cargo["itemDescription"]:
        cn_text, en_words = separate_cn_en(desc)
        if cn_text:
            all_cn_words.extend(tokenize_cn(cn_text, stopwords_cn))
        all_en_words.extend(w.upper() for w in en_words if w.upper() not in stopwords_en)

    cn_word_freq = Counter(all_cn_words).most_common(100)
    en_word_freq = Counter(all_en_words).most_common(100)
    print(f"中文词总数（条累计）: {len(all_cn_words)}")
    print(f"英文词总数（条累计）: {len(all_en_words)}")
    print("\n【中文高频词 Top 20】")
    for i, (word, freq) in enumerate(cn_word_freq[:20], 1):
        print(f"  {i:2d}. {word:20s} {freq:6d}")
    print("\n【英文高频词 Top 20】")
    for i, (word, freq) in enumerate(en_word_freq[:20], 1):
        print(f"  {i:2d}. {word:20s} {freq:6d}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    top_cn = cn_word_freq[:20]
    axes[0].barh([w[0] for w in top_cn[::-1]], [w[1] for w in top_cn[::-1]], color="steelblue", edgecolor="white")
    axes[0].set_title("中文高频词 Top 20（整合版分词）", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("词频")
    top_en = en_word_freq[:20]
    axes[1].barh([w[0] for w in top_en[::-1]], [w[1] for w in top_en[::-1]], color="coral", edgecolor="white")
    axes[1].set_title("英文高频词 Top 20（整合版分词）", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("词频")
    plt.tight_layout()
    fig.savefig(OUT / "eda_词频分析_整合版.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {OUT / 'eda_词频分析_整合版.png'}")

    # 词云（基于整合版词频）
    font_path = _pick_wordcloud_font()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    wc_cn = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        collocations=False,
        max_font_size=100,
    ).generate_from_frequencies(dict(cn_word_freq))
    axes[0].imshow(wc_cn, interpolation="bilinear")
    axes[0].set_title("中文词云（整合版）", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    if font_path is None:
        axes[0].set_title("中文词云（无中文字体路径时可能显示异常）", fontsize=12)

    wc_en = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        collocations=False,
        max_font_size=100,
    ).generate_from_frequencies(dict(en_word_freq))
    axes[1].imshow(wc_en, interpolation="bilinear")
    axes[1].set_title("英文词云", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "eda_词云_整合版.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {OUT / 'eda_词云_整合版.png'}")

    # ---------- 5. HS 与大类（Fintech0510 + 修复 merge 列） ----------
    print("\n5. HS CODE 提取与类别分布")
    df_cargo["hs_code_extracted"] = df_cargo["itemDescription"].apply(extract_hs_code)
    hs_extracted_count = df_cargo["hs_code_extracted"].notna().sum()
    print(f"从描述中提取到 HS6 的样本数: {hs_extracted_count} ({hs_extracted_count / len(df_cargo) * 100:.1f}%)")

    if hs_extracted_count > 0:
        hs_dist = df_cargo["hs_code_extracted"].value_counts().head(20)
        print("\n【提取到的 HS CODE Top 20】")
        for code, count in hs_dist.items():
            print(f"  HS{code}: {count} 条")

    if df_shcode is not None and "HSCODE" in df_shcode.columns and "大类名称" in df_shcode.columns:
        df_shcode = df_shcode.copy()
        df_shcode["HSCODE_str"] = _shcode_hs6_str_series(df_shcode)
        df_merged = df_cargo.merge(
            df_shcode[["HSCODE_str", "大类名称"]].drop_duplicates(subset="HSCODE_str"),
            left_on="hs_code_extracted",
            right_on="HSCODE_str",
            how="left",
        )
        if df_merged["大类名称"].notna().any():
            category_dist = df_merged[df_merged["大类名称"].notna()]["大类名称"].value_counts()
            print("\n【基于提取 HS 的大类分布（关联 Shcode）】")
            print(category_dist.head(20))
            fig, ax = plt.subplots(figsize=(12, 8))
            top_categories = category_dist.head(15)
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
            ax.barh(range(len(top_categories)), top_categories.values, color=colors, edgecolor="white")
            ax.set_yticks(range(len(top_categories)))
            ax.set_yticklabels(top_categories.index)
            ax.set_xlabel("样本数")
            ax.set_title("HS 关联的大类分布 Top 15", fontsize=14, fontweight="bold")
            ax.invert_yaxis()
            for i, v in enumerate(top_categories.values):
                ax.text(v + 5, i, str(v), va="center", fontsize=10)
            plt.tight_layout()
            fig.savefig(OUT / "eda_类别分布.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"已保存: {OUT / 'eda_类别分布.png'}")

    # ---------- 6. 噪声（Fintech0510） ----------
    print("\n6. 噪声类型分析")
    noise_patterns = {
        "集装箱号": r"[A-Z]{4}\d{7}",
        "包装信息": r"\d{1,4}\s*(?:PACKAGES?|PKGS?|CTNS?|CARTONS?|PALLETS?|ROLLS?|BUNDLES?)",
        "重量体积": r"\d{1,6}\.?\d*\s*(?:KGS?|KG|LBS?|CBM|CUBIC)",
        "物流参考号": r"(?:CNDC|ML-CN|S/\w|B/L|BL)\s*\d+",
        "HS_CODE": r"HS\s*CODE:?\s*\d{4,10}",
        "集装箱信息": r"(?:ONEU|TTNU|TIIU|TGBU|TCLU|FCIU)\d{7}",
        "邮箱地址": r"[\w\.-]+@[\w\.-]+\.\w+",
        "网址URL": r"https?://\S+",
        "电话传真": r"(?:TEL|FAX|PHONE|TEL/FAX)[\s:]*[\d\-\(\)\+]+",
        "金额价格": r"(?:USD|EUR|RMB|CNY|TOTAL|AMOUNT)\s*[\d,\.]+",
        "VIN码": r"VIN\s*[:\s]*[\w\d]+|LFM[A-Z0-9]{14}",
        "多行换行符": r"\n{3,}",
        "特殊字符": r"[^\u4e00-\u9fff\w\s\-\(\)\[\]\.\,\;\:\/\\\&\#\+\=\@]",
    }
    noise_stats: dict[str, int] = {}
    for noise_type, pattern in noise_patterns.items():
        count = df_cargo["itemDescription"].astype(str).str.contains(pattern, regex=True, na=False).sum()
        noise_stats[noise_type] = count
        pct = count / len(df_cargo) * 100
        print(f"  {noise_type:12s}: {count:6d} 条 ({pct:5.1f}%)")

    fig, ax = plt.subplots(figsize=(12, 6))
    noise_sorted = sorted(noise_stats.items(), key=lambda x: x[1], reverse=True)
    noise_names = [x[0] for x in noise_sorted]
    noise_values = [x[1] for x in noise_sorted]
    colors = ["#FF6B6B" if v > len(df_cargo) * 0.1 else "#4ECDC4" for v in noise_values]
    bars = ax.bar(range(len(noise_names)), noise_values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(noise_names)))
    ax.set_xticklabels(noise_names, rotation=45, ha="right")
    ax.set_ylabel("出现次数")
    ax.set_title("各类噪声模式出现频次", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, noise_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val}\n({val / len(df_cargo) * 100:.1f}%)",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    fig.savefig(OUT / "eda_噪声分析.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {OUT / 'eda_噪声分析.png'}")

    print("\n【典型噪声样本】")
    noise_samples = {
        "含集装箱号": r"[A-Z]{4}\d{7}",
        "含包装信息": r"\d{1,4}\s*PACKAGES?",
        "含HS CODE": r"HS\s*CODE:?\s*\d{4,10}",
        "含VIN码": r"LFM[A-Z0-9]{14}",
    }
    for name, pattern in noise_samples.items():
        samples = df_cargo[
            df_cargo["itemDescription"].astype(str).str.contains(pattern, regex=True, na=False)
        ]["itemDescription"].head(2)
        print(f"\n  [{name}]")
        for s in samples:
            print(f"    {str(s)[:120]}...")

    # ---------- 7. 摘要 ----------
    print("\n7. 数据质量综合评估")
    summary = {
        "总样本数": len(df_cargo),
        "唯一描述数": df_cargo["itemDescription"].nunique(),
        "重复率": f"{df_cargo['itemDescription'].duplicated().sum() / len(df_cargo) * 100:.1f}%",
        "缺失值数": df_cargo["itemDescription"].isnull().sum(),
        "平均字符长度": f"{df_cargo['char_length'].mean():.1f}",
        "中位数字符长度": f"{df_cargo['char_length'].median():.1f}",
        "中英混合比例": f"{(df_cargo['language_type'] == '中英混合').sum() / len(df_cargo) * 100:.1f}%",
        "纯中文比例": f"{(df_cargo['language_type'] == '纯中文').sum() / len(df_cargo) * 100:.1f}%",
        "纯英文比例": f"{(df_cargo['language_type'] == '纯英文').sum() / len(df_cargo) * 100:.1f}%",
        "HS CODE可提取率": f"{hs_extracted_count / len(df_cargo) * 100:.1f}%",
        "含物流信息(集装箱号)比例": f"{noise_stats.get('集装箱号', 0) / len(df_cargo) * 100:.1f}%",
        "含包装信息比例": f"{noise_stats.get('包装信息', 0) / len(df_cargo) * 100:.1f}%",
    }
    for key, value in summary.items():
        print(f"  {key:22s}: {value}")

    print("\n【主要挑战（摘自原探索脚本）】")
    for c in [
        "1. 多语言混杂：中英文混合描述占比高，需分别处理中文分词与英文词统计。",
        "2. 噪声丰富：物流单证片段多，清洗需与品类语义平衡。",
        "3. HS 覆盖有限：仅部分描述自带 HS，可作弱标签参考。",
        "4. 长度差异大：短文本分类难度大。",
        "5. 行业术语多样：同一货物多种写法。",
        "6. 混合货物：列举多种商品时存在多标签倾向。",
        "7. 模糊描述：极短描述缺乏上下文。",
    ]:
        print(f"  {c}")

    print("\n" + "=" * 60)
    print(f"Step 1 完成，图表目录: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
