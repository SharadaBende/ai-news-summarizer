"""Run all experiments from the command line and save a results report.

Examples
  python run_experiments.py                                   # built-in demo data
  python run_experiments.py --csv bbc-text.csv                # BBC News (columns: category, text)
  python run_experiments.py --csv bbc-text.csv --refs cnn_dailymail.csv --ref-text article --ref-summary highlights --max-refs 300
"""
import argparse
import json
import os

import joblib
import pandas as pd

import classifier as clf
import clustering
import data_loader as dl
import evaluation as ev


def md(df: pd.DataFrame) -> str:
    """Tiny DataFrame -> markdown table (no extra dependency)."""
    d = df.reset_index()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=dl.DEFAULT_CSV, help="classification data: text + category columns")
    ap.add_argument("--refs", default="data/sample_references.csv", help="CSV with articles + reference summaries")
    ap.add_argument("--ref-text", default="text")
    ap.add_argument("--ref-summary", default="reference")
    ap.add_argument("--max-refs", type=int, default=300)
    ap.add_argument("--sentences", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = dl.load_csv(a.csv)
    print(f"Loaded {len(df)} articles, {df['category'].nunique()} categories")

    comparison = clf.compare_models(df)
    model, m = clf.train(df)
    clus = clustering.cluster_articles(df)
    refs = dl.load_reference_csv(a.refs, a.ref_text, a.ref_summary)
    rouge = ev.evaluate(refs, n_sentences=a.sentences, max_rows=a.max_refs)

    joblib.dump(model, os.path.join(a.out, "model.joblib"))
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"holdout_accuracy": m["accuracy"], "kmeans_ari": clus["ari"],
                   "n_articles": len(df), "n_reference_articles": len(refs)}, f, indent=2)

    report = (f"# Results\n\nClassification data: `{a.csv}` ({len(df)} articles)\n\n"
              f"## Model comparison (cross-validated)\n\n{md(comparison)}\n\n"
              f"## Hold-out accuracy (final model)\n\n{m['accuracy']:.1%}\n\n"
              f"## Summarization quality (ROUGE F1, {a.sentences} sentences, {len(refs)} articles from `{a.refs}`)\n\n{md(rouge)}\n\n"
              f"## KMeans clustering vs true categories\n\nAdjusted Rand Index: {clus['ari']:.3f}\n")
    with open(os.path.join(a.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
