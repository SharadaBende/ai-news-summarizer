"""Summary-quality evaluation: ROUGE (pure Python) + baseline summarizers.

ROUGE-1/2 use unigram/bigram overlap; ROUGE-L uses the longest common subsequence.
All scores are F1 in the range 0-1. (No stemming, so numbers can differ slightly
from the official `rouge-score` package.)
"""
import random
import re
from collections import Counter

import pandas as pd

import summarizer as sm

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _TOKEN.findall(text.lower())


def _prf(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap == 0 or pred_total == 0 or ref_total == 0:
        return 0.0
    p, r = overlap / pred_total, overlap / ref_total
    return 2 * p * r / (p + r)


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(pred: str, ref: str, n: int = 1) -> float:
    p, r = _ngrams(tokenize(pred), n), _ngrams(tokenize(ref), n)
    return _prf(sum((p & r).values()), sum(p.values()), sum(r.values()))


def _lcs_len(a, b) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    a, b = tokenize(pred), tokenize(ref)
    return _prf(_lcs_len(a, b), len(a), len(b))


def rouge_scores(pred: str, ref: str) -> dict:
    return {"ROUGE-1": rouge_n(pred, ref, 1), "ROUGE-2": rouge_n(pred, ref, 2), "ROUGE-L": rouge_l(pred, ref)}


# ---------- baselines & systems (each: (text, n) -> list of sentences) ----------
def lead_n(text: str, n: int = 3):
    """Baseline: first n sentences (surprisingly strong for news)."""
    return sm.split_sentences(text)[:n]


def random_n(text: str, n: int = 3, seed: int = 0):
    """Baseline: n random sentences, kept in original order."""
    sents = sm.split_sentences(text)
    if len(sents) <= n:
        return sents
    return [sents[i] for i in sorted(random.Random(seed).sample(range(len(sents)), n))]


def textrank(text: str, n: int = 3):
    return sm.summarize(text, n)


SYSTEMS = {"Random-N (baseline)": random_n, "Lead-N (baseline)": lead_n, "TextRank (ours)": textrank}


def evaluate(df: pd.DataFrame, text_col: str = "text", ref_col: str = "reference",
             n_sentences: int = 3, systems: dict = None, max_rows: int = None) -> pd.DataFrame:
    """Average ROUGE of each system over the rows of df. Returns one row per system."""
    systems = systems or SYSTEMS
    data = df[[text_col, ref_col]].dropna()
    if max_rows:
        data = data.head(max_rows)
    rows = []
    for name, fn in systems.items():
        scores, lengths = [], []
        for text, ref in zip(data[text_col], data[ref_col]):
            summary = " ".join(fn(text, n_sentences))
            scores.append(rouge_scores(summary, ref))
            lengths.append(len(summary.split()))
        out = pd.DataFrame(scores).mean().to_dict()
        out["Avg summary words"] = sum(lengths) / max(len(lengths), 1)
        rows.append({"System": name, **out})
    return pd.DataFrame(rows).set_index("System").round(4)
