# AI News Summarizer

End-to-end NLP project: summarize news articles, classify their topic, cluster stories, and
**measure quality** - built with Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn and Streamlit.

## Features
| Area | What it does |
|---|---|
| Extractive summarization | TF-IDF sentence vectors -> cosine-similarity graph -> TextRank (PageRank implemented in NumPy) |
| Abstractive summarization (optional) | Hugging Face DistilBART via `transformer_summarizer.py` |
| Evaluation | Pure-Python **ROUGE-1/2/L**, compared against **Random** and **Lead-N** baselines |
| Classification | TF-IDF + Logistic Regression; cross-validated comparison with Naive Bayes and Linear SVM |
| Clustering | KMeans on TF-IDF, scored against true categories with Adjusted Rand Index |
| Live news | RSS fetching (BBC, NDTV, The Hindu, or any URL) |
| App | Streamlit UI with 4 tabs and Matplotlib/Seaborn charts |
| Quality | Unit tests in `tests/`, reproducible `run_experiments.py` |

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Optional extras: `pip install -r requirements-optional.txt` (transformer summarizer, ~2 GB),
`pip install -r requirements-dev.txt` (pytest).

## Reproduce the results
```bash
python run_experiments.py                       # demo data (30 articles) - illustrative only
python run_experiments.py --csv bbc-text.csv    # real classification results (BBC News, Kaggle)
python run_experiments.py --csv bbc-text.csv --refs cnn_dailymail.csv --ref-text article --ref-summary highlights --max-refs 300
python -m pytest -q                             # run tests
```
Output is written to `results/results.md` and `results/metrics.json`.

## Results
**The numbers below come from the tiny built-in demo data and are NOT representative.**
Re-run with BBC News and CNN/DailyMail, then paste your real numbers here.

| Experiment (demo data) | Result |
|---|---|
| Linear SVM / Naive Bayes / Logistic Regression (CV accuracy) | 0.93 / 0.83 / 0.80 |
| KMeans Adjusted Rand Index | 0.50 |
| ROUGE on 5 demo articles | too few articles to draw conclusions |

## Project structure
```
app.py                       Streamlit UI
summarizer.py                sentence splitting, TextRank summarizer, keywords
transformer_summarizer.py    optional abstractive model
evaluation.py                ROUGE + baselines
classifier.py                train / evaluate / compare models
clustering.py                KMeans topic discovery
data_loader.py               CSV/RSS loading and cleaning
run_experiments.py           one-command experiments + results report
tests/test_core.py           unit tests
data/                        sample_news.csv, sample_references.csv (demo)
```

## Limitations
- Extractive only (can't rewrite sentences); sentence splitting is regex-based and can break on abbreviations.
- ROUGE measures word overlap, not factual accuracy or readability; no stemming is applied.
- Demo data is tiny; real conclusions need the full datasets.

## Future work
Sentiment analysis, named-entity extraction, fine-tuning a transformer classifier, a RAG-style Q&A over articles.
