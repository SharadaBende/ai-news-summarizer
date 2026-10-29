# AI News Summarizer

An extractive news summarizer + category classifier built with Python, NumPy, Pandas,
Matplotlib, Seaborn, scikit-learn and Streamlit.

## Features
- **Extractive summarization** - TF-IDF sentence vectors, cosine-similarity graph, TextRank (PageRank in NumPy)
- **Keyword extraction** - TF-IDF unigrams/bigrams
- **Category classification** - TF-IDF + Logistic Regression, with accuracy, confusion matrix, F1
- **Live news** - fetch RSS feeds (BBC, NDTV, The Hindu, or your own URL)
- **Charts** - Matplotlib/Seaborn visuals inside the Streamlit UI

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
app.py              Streamlit UI (3 tabs)
summarizer.py       sentence splitting, TextRank summarizer, keywords
classifier.py       train / evaluate / predict category
data_loader.py      CSV loading, text cleaning, RSS fetching
make_sample_data.py generates data/sample_news.csv (30 starter articles)
data/sample_news.csv
```

## Using a real dataset
Download the BBC News dataset from Kaggle, make sure it has columns named `text` and `category`
(rename if needed), and upload it from the sidebar or replace `data/sample_news.csv`.
The classifier will reach roughly 95%+ accuracy on it.

## Ideas to extend
ROUGE evaluation, sentiment analysis, KMeans clustering of similar stories, word clouds, PDF export.
