"""Loading and cleaning news data (CSV dataset or live RSS feeds)."""
import re
import html
import pandas as pd

DEFAULT_CSV = "data/sample_news.csv"

RSS_FEEDS = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "NDTV Top Stories": "https://feeds.feedburner.com/ndtvnews-top-stories",
    "The Hindu": "https://www.thehindu.com/news/national/feed/",
}


def clean_text(text: str) -> str:
    """Remove HTML tags, entities, URLs and extra whitespace."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_csv(path_or_buffer=DEFAULT_CSV, text_col="text", label_col="category") -> pd.DataFrame:
    """Load a CSV that has at least a text column and a category column."""
    df = pd.read_csv(path_or_buffer)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns. Found: {list(df.columns)}")
    df = df.rename(columns={text_col: "text", label_col: "category"})
    df = df.dropna(subset=["text", "category"]).drop_duplicates(subset="text")
    df["text"] = df["text"].map(clean_text)
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["word_count"] = df["text"].str.split().str.len()
    return df.reset_index(drop=True)


def fetch_rss(url: str, limit: int = 15) -> pd.DataFrame:
    """Fetch articles from an RSS feed. Needs `feedparser` and internet access."""
    try:
        import feedparser
    except ImportError as e:
        raise RuntimeError("feedparser is not installed. Run: pip install feedparser") from e

    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:limit]:
        body = entry.get("content", [{}])[0].get("value") or entry.get("summary", "")
        rows.append({
            "title": clean_text(entry.get("title", "")),
            "text": clean_text(body),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
        })
    return pd.DataFrame(rows)


def load_reference_csv(path_or_buffer, text_col: str = "text", ref_col: str = "reference") -> pd.DataFrame:
    """Load articles with human reference summaries (e.g. CNN/DailyMail: article, highlights)."""
    df = pd.read_csv(path_or_buffer)
    missing = [c for c in (text_col, ref_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")
    df = df.rename(columns={text_col: "text", ref_col: "reference"}).dropna(subset=["text", "reference"])
    df["text"] = df["text"].map(clean_text)
    df["reference"] = df["reference"].map(clean_text)
    return df.reset_index(drop=True)
