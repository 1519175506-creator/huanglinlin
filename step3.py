#!/usr/bin/env python3
"""
FINA2003 模块（二）2.3：L1 HS 抽取 → L2 TF-IDF 余弦最近邻（海关知识库）→ L3 关键词兜底。

可被 cargo_classification_analysis.py 调用 run_classification_module；
亦可独立运行（读项目根目录 `货物描述信息2025.xlsx`、`海关shcode表.xlsx`，写出 output/）。
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent
PATH_CARGO = PROJECT_ROOT / "货物描述信息2025.xlsx"
PATH_SHCODE = PROJECT_ROOT / "海关shcode表.xlsx"
DEFAULT_OUT = PROJECT_ROOT / "output"

# L2 余弦相似度低于该阈值时触发 L3（关键词兜底）
L2_SIM_THRESHOLD = 0.15
L3_MIN_SUPPORT = 5
COSINE_BATCH = 512
# 知识库「大类/细类」刻板表述，作关键词会误导 HS6 众数
L3_KB_ZH_NOISE = set(
    "及其 制品 其他 附件 设备 原料 类似 零件 上述 所述 包括 不含 产品 用品 货物 商品 "
    "未列名 等相关 等其他 或其他 及其他 活动物 贱金属 塑料 橡胶 机器 机械 器具".split()
)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------------------------------------------------------
# 文本预处理（与分类共用）
# -----------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def token_has_chinese(tok: str) -> bool:
    """判断分词单元是否含中日韩统一表意文字（本作业语境下即「中文词」）。"""
    return _CJK_RE.search(tok) is not None


def series_has_cjk(series: pd.Series) -> pd.Series:
    """条级别：描述是否至少含一个汉字。"""
    return series.astype(str).str.contains(_CJK_RE, regex=True)


_CONTAINER_NOISE_RE = re.compile(
    r"\b\d+[A-Z]{3}\d{7}\b|"
    r"\b[A-Z]{4}\d{7}\b|"
    r"/[A-Z]*\d+[A-Z]*/|"
    r"\d+\s*(?:KGS|KG|CBM|PACKAGES)\b",
    re.I,
)
_NUMERIC_RUN_RE = re.compile(r"\d[\d,\.\s]*")


def preprocess_text(raw: str) -> str:
    """
    1) Unicode NFC：合并兼容字符，避免同一字形多种编码导致词表碎片化。
    2) 统一换行/制表符为空格：货品描述常含提单式换行与 \\t，若不展开会破坏分词与 n-gram。
    3) 压缩空白：避免多余空格影响 TF-IDF。
    4) 弱化航运单证噪声：集装箱号、毛重体积等对「商品类别」判别贡献低，却占用极高 IDF，
       故做保守剔除（保留中文品名主体）。
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = unicodedata.normalize("NFC", str(raw))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ")
    s = _CONTAINER_NOISE_RE.sub(" ", s)
    s = _NUMERIC_RUN_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def zh_tokens(text: str):
    """jieba 精确模式：中文保险/海关场景下速度与可控性平衡较好。"""
    text = preprocess_text(text)
    if not text:
        return []
    return [t.strip() for t in jieba.lcut(text) if t.strip()]


def zh_tokens_from_clean(clean: str) -> list[str]:
    """输入已为 preprocess_text 后的字符串，避免重复清洗。"""
    if not clean:
        return []
    return [t.strip() for t in jieba.lcut(clean) if t.strip()]


def space_joined_tokens_from_clean(clean: str) -> str:
    """供 TF-IDF 按空白切分：避免 Vectorizer 每条样本反复调用 jieba。"""
    return " ".join(zh_tokens_from_clean(clean))


def parallel_space_joined(clean_series: pd.Series, max_workers: int = 8) -> list[str]:
    """线程并行分词（jieba C 扩展释放 GIL，线程通常有收益）。"""
    texts = clean_series.tolist()
    n = len(texts)
    if n < 500:
        return [space_joined_tokens_from_clean(s) for s in texts]
    out = [None] * n
    chunk_size = max(400, n // max_workers)

    def chunk_job(start: int):
        end = min(n, start + chunk_size)
        return start, [space_joined_tokens_from_clean(t) for t in texts[start:end]]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(chunk_job, st) for st in range(0, n, chunk_size)]
        for fut in futures:
            start, rows = fut.result()
            for i, row in enumerate(rows):
                out[start + i] = row
    return out


# -----------------------------------------------------------------------------
# EDA 辅助（L3 规则构建与主脚本探索共用）
# -----------------------------------------------------------------------------
_ZH_STOP = set(
    "的 及 和 与 或 等 若干 各类 各种 用于 以上 以下 包装 规格 型号 产品 货物 商品 "
    "材料 部件 零件 配件 名称 品名".split()
)


def _count_tokens_filtered(series: pd.Series, keep_token) -> Counter:
    cnt: Counter = Counter()
    for raw in series.dropna():
        for w in zh_tokens(str(raw)):
            if len(w) < 2:
                continue
            if not keep_token(w):
                continue
            cnt[w] += 1
    return cnt


def eda_word_freq(series: pd.Series, topn: int = 40) -> pd.Series:
    """全部分词后的高频词（中英混合，易呈现英文型号主导）。"""
    cnt = _count_tokens_filtered(series, lambda _: True)
    for w in list(cnt.keys()):
        if w in _ZH_STOP:
            del cnt[w]
    most = cnt.most_common(topn)
    return pd.Series({w: c for w, c in most}, name="频次")


def eda_word_freq_chinese(series: pd.Series, topn: int = 40) -> pd.Series:
    """仅统计含汉字的分词单元。"""
    cnt = _count_tokens_filtered(series, token_has_chinese)
    for w in list(cnt.keys()):
        if w in _ZH_STOP:
            del cnt[w]
    most = cnt.most_common(topn)
    return pd.Series({w: c for w, c in most}, name="频次")


# -----------------------------------------------------------------------------
# L1 / L2 / L3
# -----------------------------------------------------------------------------
_HS_DOT = re.compile(
    r"\b(\d{4})\s*[\.\/\-]\s*(\d{2})\s*[\.\/\-]\s*(\d{2})\s*[\.\/\-]\s*(\d{2})\b"
)
_HS10 = re.compile(r"\b(\d{10})\b")
_HS8 = re.compile(r"\b(\d{8})\b")
_HS_PREFIX = re.compile(
    r"(?:HS|H\.S\.|HSCODE|税号|海关编码)\s*[:：]?\s*(\d{6,12})\b",
    re.I,
)


def hscode_value_to_hs6(val) -> str | None:
    """将海关表中的 HSCODE（常为浮点）规范为 6 位 HS（税号前六位）。"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if isinstance(val, (int, np.integer)):
            d = str(int(val))
        else:
            s = str(val).strip()
            if not s or s.lower() == "nan":
                return None
            if re.fullmatch(r"-?\d+(\.\d+)?", s):
                d = re.sub(r"\D", "", str(int(float(s))))
            else:
                d = re.sub(r"\D", "", s)
    except (ValueError, OverflowError):
        d = re.sub(r"\D", "", str(val))
    if len(d) >= 10:
        d = d[:10]
    if len(d) >= 6:
        return d[:6]
    return None


def extract_hs6_l1(text: str) -> str | None:
    """
    L1：从原始货品描述中直接抽取 HS / 税号片段 → 取前 6 位。
    须在 preprocess 之前或并行使用原始字段，避免数字被清洗规则削弱。
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return None
    m = _HS_PREFIX.search(s)
    if m:
        d = re.sub(r"\D", "", m.group(1))
        if len(d) >= 6:
            return d[:6]
    m = _HS_DOT.search(s)
    if m:
        d = "".join(m.groups())
        if len(d) >= 6:
            return d[:6]
    m = _HS10.search(s)
    if m:
        return m.group(1)[:6]
    m = _HS8.search(s)
    if m:
        return m.group(1)[:6]
    return None


def build_kb_from_shcode(sh: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    知识库：每条有效记录 = 规范化品名（预处理+jieba 词串）→ HS6 + 大类。
    pin_raw 保留原始品名字符串供 L3 子串匹配。
    """
    kb_toks: list[str] = []
    kb_hs6: list[str] = []
    kb_major: list[str] = []
    kb_pin_raw: list[str] = []
    for _, r in sh.iterrows():
        h6 = hscode_value_to_hs6(r["HSCODE"])
        if not h6:
            continue
        pin_raw = str(r["品名"]).strip()
        pm = preprocess_text(pin_raw)
        if not pm:
            continue
        tok = space_joined_tokens_from_clean(pm)
        if not tok.strip():
            continue
        kb_toks.append(tok)
        kb_hs6.append(h6)
        kb_major.append(str(r["大类名称"]))
        kb_pin_raw.append(pin_raw)
    return kb_toks, kb_hs6, kb_major, kb_pin_raw


def hs6_to_major_map(sh: pd.DataFrame) -> dict[str, str]:
    tmp = sh.assign(hs6=sh["HSCODE"].map(hscode_value_to_hs6)).dropna(subset=["hs6"])
    return tmp.groupby("hs6")["大类名称"].agg(lambda x: x.mode().iloc[0]).to_dict()


def make_tfidf_vectorizer_kb() -> TfidfVectorizer:
    return TfidfVectorizer(
        token_pattern=r"(?u)\S+",
        analyzer="word",
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=35000,
    )


def batch_argmax_cosine_similarity(X_q, X_kb, batch: int = COSINE_BATCH):
    """对每条查询返回与知识库余弦相似度最大的下标及该相似度。"""
    n = X_q.shape[0]
    best_idx = np.empty(n, dtype=np.int64)
    best_sim = np.empty(n)
    for start in range(0, n, batch):
        end = min(n, start + batch)
        sims = cosine_similarity(X_q[start:end], X_kb)
        j = sims.argmax(axis=1)
        best_idx[start:end] = j
        best_sim[start:end] = sims[np.arange(end - start), j]
    return best_idx, best_sim


def kb_concat_text_per_row(sh: pd.DataFrame) -> pd.Series:
    """知识库侧文本：品名 + 细类 + 大类，经与货描相同的 preprocess。"""
    pin = sh["品名"].astype(str)
    xi = sh["细类名称"].astype(str)
    da = sh["大类名称"].astype(str)
    out: list[str] = []
    for a, b, c in zip(pin, xi, da, strict=True):
        out.append(preprocess_text(f"{a} {b} {c}"))
    return pd.Series(out, index=sh.index)


def _l3_append_from_freq(
    rules: list[tuple[str, str, bool, str]],
    seen: set[tuple[str, bool]],
    freq_en: pd.Series,
    freq_zh: pd.Series,
    sh: pd.DataFrame,
    match_text: pd.Series,
    top_en: int,
    top_zh: int,
    source: str,
) -> None:
    """在 match_text 含该词的记录上取 HS6 众数；source 标记 kb / cargo。"""
    for w in list(freq_en.index)[:top_en]:
        if not re.fullmatch(r"[A-Za-z]{3,}", str(w)):
            continue
        kw = str(w).upper()
        key = (kw, True)
        if key in seen:
            continue
        sub = sh.loc[match_text.str.contains(re.escape(kw), case=False, na=False)]
        if len(sub) < L3_MIN_SUPPORT:
            continue
        hs6s = sub["HSCODE"].map(hscode_value_to_hs6).dropna()
        if hs6s.empty:
            continue
        rules.append((kw, str(hs6s.mode().iloc[0]), True, source))
        seen.add(key)

    for w in list(freq_zh.index)[:top_zh]:
        kw = str(w)
        if len(kw) < 2:
            continue
        if source == "kb":
            if len(kw) < 3 or kw in L3_KB_ZH_NOISE:
                continue
        key = (kw, False)
        if key in seen:
            continue
        sub = sh.loc[match_text.str.contains(re.escape(kw), na=False)]
        if len(sub) < L3_MIN_SUPPORT:
            continue
        hs6s = sub["HSCODE"].map(hscode_value_to_hs6).dropna()
        if hs6s.empty:
            continue
        rules.append((kw, str(hs6s.mode().iloc[0]), False, source))
        seen.add(key)


def build_l3_keyword_rules(
    sh: pd.DataFrame,
    cargo_desc_raw: pd.Series,
    cargo_top_en: int = 70,
    cargo_top_zh: int = 45,
    kb_top_en: int = 90,
    kb_top_zh: int = 70,
) -> list[tuple[str, str, bool, str]]:
    """
    L3 关键词规则（两路合并）：
    1）知识库侧词频 → 支持度内 HS6 众数；
    2）货描侧词频 → 在海关「品名」中共现 ≥ 支持度 → HS6 众数。
    """
    rules: list[tuple[str, str, bool, str]] = []
    seen: set[tuple[str, bool]] = set()

    kb_text = kb_concat_text_per_row(sh)
    freq_en_kb = eda_word_freq(kb_text, topn=260)
    freq_zh_kb = eda_word_freq_chinese(kb_text, topn=180)
    _l3_append_from_freq(
        rules,
        seen,
        freq_en_kb,
        freq_zh_kb,
        sh,
        kb_text,
        kb_top_en,
        kb_top_zh,
        "kb",
    )

    pm = sh["品名"].astype(str)
    freq_en_c = eda_word_freq(cargo_desc_raw, topn=160)
    freq_zh_c = eda_word_freq_chinese(cargo_desc_raw, topn=100)
    _l3_append_from_freq(
        rules,
        seen,
        freq_en_c,
        freq_zh_c,
        sh,
        pm,
        cargo_top_en,
        cargo_top_zh,
        "cargo",
    )

    rules.sort(key=lambda t: len(t[0]), reverse=True)
    return rules


def apply_l3_keyword(desc_raw: str, keyword_rules: list[tuple[str, str, bool, str]]) -> str | None:
    raw_up = str(desc_raw).upper()
    raw_plain = str(desc_raw)
    for kw, hs6, english_only, *_ in keyword_rules:
        if english_only:
            if kw in raw_up:
                return hs6
        else:
            if kw in raw_plain:
                return hs6
    return None


def pred_major_from_hs6(hs6: str | None, hs6_major: dict[str, str]) -> str:
    if hs6 and hs6 in hs6_major:
        return hs6_major[hs6]
    return "特殊交易品及未分类商品"


def plot_confusion_hs6(y_true, y_pred, path: Path, max_labels: int = 15):
    """HS6 类别极多，仅展示真实标签中出现频次最高的若干类的混淆子矩阵。"""
    ctr = Counter(y_true)
    tick_labels = [lab for lab, _ in ctr.most_common(max_labels)]
    cm = confusion_matrix(y_true, y_pred, labels=tick_labels)
    k = len(tick_labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"HS6 混淆矩阵（按真实频次 Top {k}）")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("真实 HS6")
    ax.set_xlabel("预测 HS6")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def describe_noise(series: pd.Series) -> pd.DataFrame:
    s = series.astype(str)

    def rate(pat: str) -> float:
        return float(s.str.contains(pat, regex=True).mean())

    rows = [
        ("含换行符 \\n", rate(r"\n")),
        ("含制表符 \\t", rate(r"\t")),
        ("中英混排（含 ASCII 字母）", rate(r"[A-Za-z]")),
        ("连续数字片段", rate(r"\d{4,}")),
        ("疑似集装箱/单证模板片段", rate(r"\b\d+[A-Z]{3}\d{7}\b|[A-Z]{4}\d{7}|KGS|CBM|PACKAGES")),
    ]
    return pd.DataFrame(rows, columns=["噪声/形态", "占比"])


def token_language_summary(series: pd.Series) -> dict[str, float | int]:
    """全库分词条级别：中文相关词占比。"""
    total = 0
    zh_tok = 0
    for raw in series.dropna():
        for w in zh_tokens(str(raw)):
            if not w:
                continue
            total += 1
            if token_has_chinese(w):
                zh_tok += 1
    ratio = float(zh_tok / total) if total else 0.0
    return {"分词单元总数": total, "含汉字单元数": zh_tok, "含汉字单元占比": ratio}


def save_wordfreq_csv(freq: pd.Series, path: Path) -> None:
    out = freq.rename("频次").reset_index()
    out.columns = ["词语", "频次"]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def plot_length_hist(lengths: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths.clip(upper=lengths.quantile(0.99)), bins=50, color="#4472c4", edgecolor="white")
    ax.set_title("货品描述字符长度分布（原始字符串，截断至 99% 分位以便展示）")
    ax.set_xlabel("字符数")
    ax.set_ylabel("频数")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_word_bar(freq: pd.Series, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    freq.iloc[::-1].plot(kind="barh", ax=ax, color="#ed7d31")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_category_bar(counts: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    counts.iloc[::-1].plot(kind="barh", ax=ax, color="#70ad47")
    ax.set_title("海关表训练标签：大类分布（条数）")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_classification_module(cargo: pd.DataFrame, sh: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    在已含 desc_raw、desc_clean 的 cargo 上运行 L1/L2/L3，写入评估与预测相关 CSV/图，
    并写出 cargo_with_predictions.csv（供 step3.py 使用）。

    返回带 pred_hs6、match_layer、l2_cosine_sim、pred_major、l1_hs6 等列的 cargo。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    kb_toks, kb_hs6, _, kb_pin_raw = build_kb_from_shcode(sh)
    hs6_major = hs6_to_major_map(sh)
    uniq_hs6 = len(set(kb_hs6))
    print("【知识库】海关 Shcode 有效条目（有 HS6 且品名可预处理）")
    print(f"- 条数: {len(kb_toks)}，唯一 HS6（前六位）类数: {uniq_hs6}\n")

    print("【进度】构建 L3 关键词规则（知识库侧词频 + 货描侧词频）…")
    l3_rules = build_l3_keyword_rules(sh, cargo["desc_raw"])
    n_kb_r = sum(1 for r in l3_rules if r[3] == "kb")
    n_cargo_r = sum(1 for r in l3_rules if r[3] == "cargo")
    print(f"【L3 规则构成】知识库侧 {n_kb_r} 条，货描侧 {n_cargo_r} 条（去重后合计 {len(l3_rules)}）\n")
    pd.DataFrame(
        l3_rules[:220],
        columns=["keyword", "hs6", "english_only", "source"],
    ).to_csv(
        out_dir / "l3_keyword_rules_preview.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n_kb = len(kb_toks)
    idx_all = np.arange(n_kb)
    tr_idx, te_idx = train_test_split(idx_all, test_size=0.2, random_state=42)
    vec_eval = make_tfidf_vectorizer_kb()
    X_kb_tr = vec_eval.fit_transform([kb_toks[i] for i in tr_idx])
    X_te_q = vec_eval.transform([kb_toks[i] for i in te_idx])
    best_loc, best_sim = batch_argmax_cosine_similarity(X_te_q, X_kb_tr)

    pred_hs6_eval: list[str] = []
    layers_eval: list[str] = []
    for i in range(len(te_idx)):
        sim = float(best_sim[i])
        global_j = int(tr_idx[best_loc[i]])
        base_hs6 = kb_hs6[global_j]
        pin = kb_pin_raw[te_idx[i]]
        layer = "L2"
        pred = base_hs6
        if sim < L2_SIM_THRESHOLD:
            l3h = apply_l3_keyword(pin, l3_rules)
            if l3h:
                pred = l3h
                layer = "L3"
            else:
                layer = "L2_lowconf"
        pred_hs6_eval.append(pred)
        layers_eval.append(layer)

    y_true_eval = [kb_hs6[i] for i in te_idx]
    acc = accuracy_score(y_true_eval, pred_hs6_eval)
    macro_f1 = f1_score(y_true_eval, pred_hs6_eval, average="macro", zero_division=0)

    print("【分类方案】L1（描述内 HS 抽取）+ L2（TF-IDF 余弦相似度最近邻）+ L3（相似度<阈值时关键词映射 HS6）")
    print(
        f"【参数】L2 置信阈值（余弦相似度）={L2_SIM_THRESHOLD}；"
        f"L3 规则={len(l3_rules)}（kb {n_kb_r} + cargo {n_cargo_r}，支持度≥{L3_MIN_SUPPORT}）\n"
    )
    print(f"Held-out（海关「品名」→ 知识库子集检索）HS6 准确率: {acc:.4f}")
    print(f"Held-out HS6 macro-F1: {macro_f1:.4f}")
    print("【验证集各层调用占比】")
    print(pd.Series(layers_eval).value_counts().to_string())
    print()

    top_labels = [lab for lab, _ in Counter(y_true_eval).most_common(25)]
    print("【分类报告（真实标签频次 Top25 HS6）】")
    print(
        classification_report(
            y_true_eval,
            pred_hs6_eval,
            labels=top_labels,
            zero_division=0,
        )
    )

    plot_confusion_hs6(y_true_eval, pred_hs6_eval, out_dir / "confusion_hs6_top15.png", max_labels=15)

    errors = pd.DataFrame(
        {
            "y_true_hs6": y_true_eval,
            "y_pred_hs6": pred_hs6_eval,
            "layer": layers_eval,
            "cosine_sim": list(best_sim),
            "品名样本": [kb_pin_raw[i] for i in te_idx],
        }
    )
    errors = errors[errors["y_true_hs6"] != errors["y_pred_hs6"]]
    errors.head(40).to_csv(out_dir / "error_cases_sample.csv", index=False, encoding="utf-8-sig")

    print("【错误分析 — 归纳】")
    print(
        "- HS6 粒度细（唯一类远多于 22 个大类），最近邻在「品名」极短或通用词多时易串类。\n"
        "- L2_lowconf：余弦低于阈值且 L3 未命中，仍保留最近邻 HS6，对应报告中的「勉强匹配」风险。\n"
        "- 改进：引入 HS 章节先验、字符 n-gram、人工审计高分歧义对等。\n"
    )

    vec_full = make_tfidf_vectorizer_kb()
    X_kb_full = vec_full.fit_transform(kb_toks)

    cargo = cargo.copy()
    cargo["l1_hs6"] = cargo["desc_raw"].map(extract_hs6_l1)
    print("【进度】货品描述分词（L2/L3）…")
    cargo_tok = parallel_space_joined(cargo["desc_clean"])
    X_cargo = vec_full.transform(cargo_tok)
    best_idx_c, best_sim_c = batch_argmax_cosine_similarity(X_cargo, X_kb_full)

    pred_hs6_list: list[str] = []
    layer_list: list[str] = []
    sim_list: list[float] = []
    for i in range(len(cargo)):
        raw = cargo["desc_raw"].iloc[i]
        l1v = cargo["l1_hs6"].iloc[i]
        if pd.notna(l1v) and str(l1v).strip():
            pred_hs6_list.append(str(l1v))
            layer_list.append("L1")
            sim_list.append(1.0)
            continue
        j = int(best_idx_c[i])
        sim = float(best_sim_c[i])
        base = kb_hs6[j]
        pred = base
        layer = "L2"
        if sim < L2_SIM_THRESHOLD:
            l3h = apply_l3_keyword(raw, l3_rules)
            if l3h:
                pred = l3h
                layer = "L3"
            else:
                layer = "L2_lowconf"
        pred_hs6_list.append(pred)
        layer_list.append(layer)
        sim_list.append(sim)

    cargo["pred_hs6"] = pred_hs6_list
    cargo["match_layer"] = layer_list
    cargo["l2_cosine_sim"] = sim_list
    cargo["pred_major"] = [pred_major_from_hs6(h, hs6_major) for h in pred_hs6_list]

    pd.Series(layer_list).value_counts().to_csv(out_dir / "cargo_layer_counts.csv", encoding="utf-8-sig")
    cargo[["desc_raw", "pred_hs6", "pred_major", "match_layer", "l2_cosine_sim"]].head(500).to_csv(
        out_dir / "cargo_pred_sample500.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("【货描全量流水线 — 各层占比】")
    print(pd.Series(layer_list).value_counts().to_string())
    print()

    cargo[["desc_raw", "pred_hs6", "pred_major", "match_layer", "l2_cosine_sim"]].to_csv(
        out_dir / "cargo_with_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"【2.3 分类】预测结果已写入: {out_dir / 'cargo_with_predictions.csv'}")
    return cargo


def main():
    print("=== FINA2003 模块（二）2.3 分类（独立运行）===\n")
    _mpl = DEFAULT_OUT / ".mplconfig"
    _mpl.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl))
    DEFAULT_OUT.mkdir(exist_ok=True)

    try:
        jieba.enable_parallel(4)
    except Exception:
        pass

    if not PATH_CARGO.is_file():
        print(f"未找到货描文件: {PATH_CARGO}")
        return
    if not PATH_SHCODE.is_file():
        print(f"未找到海关表: {PATH_SHCODE}")
        return

    cargo = pd.read_excel(PATH_CARGO, sheet_name="数据")
    sh = pd.read_excel(PATH_SHCODE)
    desc_col = "货物描述itemDescription"
    if desc_col not in cargo.columns:
        print(f"货描表中缺少列: {desc_col}")
        return
    cargo["desc_raw"] = cargo[desc_col].astype(str)
    cargo["desc_clean"] = cargo["desc_raw"].map(preprocess_text)

    run_classification_module(cargo, sh, DEFAULT_OUT)
    print("\n模块三可执行: python3 step3.py（需同目录下 保险除外标的表.xlsx）")


if __name__ == "__main__":
    main()
