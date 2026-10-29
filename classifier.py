"""News category classifier: TF-IDF + Logistic Regression."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """Train and evaluate. Returns (model, metrics dict)."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=test_size, random_state=seed, stratify=df["category"])
    model = build_pipeline().fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = sorted(df["category"].unique())
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "report": pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T,
        "confusion": confusion_matrix(y_test, pred, labels=labels),
        "labels": labels,
        "n_train": len(X_train), "n_test": len(X_test),
    }
    return model, metrics


def predict(model, text: str):
    """Return (best_label, {label: probability})."""
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    return classes[int(np.argmax(probs))], dict(zip(classes, probs))


def top_keywords_per_class(model, n: int = 8) -> pd.DataFrame:
    """Most predictive words for each category (from logistic regression weights)."""
    terms = np.array(model.named_steps["tfidf"].get_feature_names_out())
    clf = model.named_steps["clf"]
    rows = {}
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i] if len(clf.classes_) > 2 else clf.coef_[0] * (1 if i == 1 else -1)
        rows[cls] = list(terms[np.argsort(coefs)[::-1][:n]])
    return pd.DataFrame(rows)
