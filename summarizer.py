"""Extractive summarizer: TF-IDF sentence vectors + TextRank (PageRank) scoring."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str, min_words: int = 5):
    sents = [s.strip() for s in _SPLIT.split(text.strip())]
    return [s for s in sents if len(s.split()) >= min_words]


def _pagerank(sim: np.ndarray, d: float = 0.85, iters: int = 100, tol: float = 1e-6):
    n = sim.shape[0]
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    M = sim / row_sums
    r = np.full(n, 1 / n)
    for _ in range(iters):
        new = (1 - d) / n + d * M.T @ r
        if np.abs(new - r).sum() < tol:
            break
        r = new
    return r


def summarize(text: str, n_sentences: int = 3, return_scores: bool = False):
    """Return the top n sentences (in original order). Optionally also all sentence scores."""
    sents = split_sentences(text)
    if len(sents) <= n_sentences:
        return (sents, np.ones(len(sents))) if return_scores else sents

    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    scores = _pagerank(sim)

    # small boost for earlier sentences (news puts key info first)
    position_bonus = np.linspace(0.15, 0, len(sents))
    final = scores / scores.max() + position_bonus

    top = sorted(np.argsort(final)[-n_sentences:])
    summary = [sents[i] for i in top]
    return (summary, final) if return_scores else summary


def extract_keywords(text: str, top_n: int = 8):
    """Top keywords/bigrams of a single text using TF-IDF over its sentences."""
    sents = split_sentences(text, min_words=3) or [text]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=500)
    X = vec.fit_transform(sents)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    return list(terms[np.argsort(scores)[::-1][:top_n]])


def compression_stats(text: str, summary_sents):
    orig = len(text.split())
    summ = len(" ".join(summary_sents).split())
    return {"original_words": orig, "summary_words": summ,
            "reduction_pct": round(100 * (1 - summ / orig), 1) if orig else 0.0}
