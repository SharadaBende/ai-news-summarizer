"""Unsupervised topic discovery: KMeans on TF-IDF vectors + 2-D projection for plotting."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score


def cluster_articles(df: pd.DataFrame, k: int = None, seed: int = 42, top_terms: int = 6) -> dict:
    vec = TfidfVectorizer(stop_words="english", max_df=0.9)
    X = vec.fit_transform(df["text"])
    k = k or df["category"].nunique()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)

    terms = np.array(vec.get_feature_names_out())
    cluster_terms = {f"Cluster {i}": list(terms[np.argsort(c)[::-1][:top_terms]])
                     for i, c in enumerate(km.cluster_centers_)}
    coords = TruncatedSVD(n_components=2, random_state=seed).fit_transform(X)
    return {
        "labels": km.labels_,
        "coords": coords,
        "ari": adjusted_rand_score(df["category"], km.labels_),   # 1.0 = clusters match true categories
        "cluster_terms": pd.DataFrame(cluster_terms),
    }
