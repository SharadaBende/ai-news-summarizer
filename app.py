"""AI News Summarizer - Streamlit app.  Run:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import classifier as clf
import clustering
import data_loader as dl
import evaluation as ev
import summarizer as sm
import transformer_summarizer as ts

st.set_page_config(page_title="AI News Summarizer", page_icon="📰", layout="wide")
sns.set_theme(style="whitegrid")


# ---------- cached helpers ----------
@st.cache_data
def get_dataset(uploaded=None):
    return dl.load_csv(uploaded if uploaded is not None else dl.DEFAULT_CSV)


@st.cache_resource
def get_model(df: pd.DataFrame):
    return clf.train(df)


@st.cache_data(show_spinner="Comparing models…")
def get_comparison(df: pd.DataFrame):
    return clf.compare_models(df)


@st.cache_data(show_spinner="Clustering…")
def get_clusters(df: pd.DataFrame):
    return clustering.cluster_articles(df)


@st.cache_data(ttl=600, show_spinner=False)
def get_feed(url: str, limit: int):
    return dl.fetch_rss(url, limit)


# ---------- sidebar ----------
st.sidebar.title("📰 AI News Summarizer")
method = st.sidebar.radio("Summarization method",
                          ["Extractive (TextRank)", "Abstractive (Transformer)"],
                          help="Abstractive needs `pip install -r requirements-optional.txt` and internet for the first download.")
n_sent = st.sidebar.slider("Summary length (sentences)", 1, 8, 3)
n_kw = st.sidebar.slider("Number of keywords", 3, 15, 8)
st.sidebar.markdown("---")
st.sidebar.subheader("Training data")
upload = st.sidebar.file_uploader("Upload CSV (columns: text, category)", type="csv")
try:
    df = get_dataset(upload)
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()
st.sidebar.caption(f"{len(df)} articles • {df['category'].nunique()} categories")

model, metrics = get_model(df)

tab1, tab2, tab3, tab4 = st.tabs(["✍️ Summarize", "📡 Live news (RSS)",
                                  "🧪 Summary evaluation (ROUGE)", "📊 Model & data insights"])


# ---------- shared renderer ----------
def show_result(text: str):
    label, probs = clf.predict(model, text)
    scores, summary = None, None

    if method.startswith("Abstractive"):
        if not ts.is_available():
            st.warning("Transformers/torch not installed - falling back to extractive. "
                       "Run `pip install -r requirements-optional.txt` to enable this option.")
        else:
            try:
                with st.spinner("Running transformer model (first run downloads it)…"):
                    abstract = ts.summarize_abstractive(text)
                summary = [abstract]
            except Exception as e:
                st.error(f"Transformer failed ({e}). Falling back to extractive.")
    if summary is None:
        summary, scores = sm.summarize(text, n_sent, return_scores=True)

    stats = sm.compression_stats(text, summary)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted category", label.title())
    c2.metric("Original words", stats["original_words"])
    c3.metric("Summary words", stats["summary_words"])
    c4.metric("Reduction", f"{stats['reduction_pct']}%")

    st.subheader("Summary")
    for s in summary:
        st.markdown(f"- {s}")
    st.markdown("**Keywords:** " + " ".join(f"`{k}`" for k in sm.extract_keywords(text, n_kw)))

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3))
        p = pd.Series(probs).sort_values()
        sns.barplot(x=p.values, y=p.index, ax=ax, color="#4C78A8")
        ax.set_xlabel("Probability"); ax.set_ylabel(""); ax.set_title("Category confidence")
        st.pyplot(fig)
    with col_b:
        sents = sm.split_sentences(text)
        if scores is not None and len(scores) == len(sents) and len(sents) > 1:
            fig, ax = plt.subplots(figsize=(5, 3))
            chosen = set(summary)
            ax.bar(range(1, len(sents) + 1), scores, color=["#F58518" if s in chosen else "#BBBBBB" for s in sents])
            ax.set_xlabel("Sentence #"); ax.set_ylabel("Importance")
            ax.set_title("Sentence importance (orange = in summary)")
            st.pyplot(fig)


# ---------- tab 1 ----------
with tab1:
    st.header("Paste an article")
    samples = df.groupby("category").head(1)
    choice = st.selectbox("…or load a sample article", ["(none)"] + list(samples["category"].str.title()))
    default = ""
    if choice != "(none)":
        default = samples[samples["category"] == choice.lower()]["text"].iloc[0]
    text = st.text_area("Article text", value=default, height=220,
                        placeholder="Paste a news article here (at least a few sentences)…")
    if st.button("Summarize", type="primary", key="btn_text"):
        if len(sm.split_sentences(text)) < 2:
            st.warning("Please paste a longer article (at least 2 full sentences).")
        else:
            show_result(dl.clean_text(text))


# ---------- tab 2 ----------
with tab2:
    st.header("Summarize live headlines")
    st.caption("Needs internet access and the `feedparser` package.")
    feed_name = st.selectbox("News feed", list(dl.RSS_FEEDS))
    custom = st.text_input("…or paste your own RSS URL (optional)")
    limit = st.slider("Articles to fetch", 5, 30, 10)
    if st.button("Fetch news", key="btn_rss"):
        try:
            st.session_state["feed_df"] = get_feed(custom or dl.RSS_FEEDS[feed_name], limit)
        except Exception as e:
            st.error(f"Could not fetch feed: {e}")
    feed_df = st.session_state.get("feed_df")
    if feed_df is not None and not feed_df.empty:
        idx = st.selectbox("Pick an article", range(len(feed_df)), format_func=lambda i: feed_df.loc[i, "title"])
        row = feed_df.loc[idx]
        st.markdown(f"**{row['title']}**  \n{row['published']}  \n[Open original]({row['link']})")
        body = row["text"]
        if len(sm.split_sentences(body)) < 2:
            st.info("This feed only provides a headline/short blurb, so there isn't enough text to summarize. "
                    "Try a different feed or paste the full article in the first tab.")
            st.write(body)
        else:
            show_result(body)
    elif feed_df is not None:
        st.info("No articles returned for that feed.")


# ---------- tab 3: ROUGE ----------
with tab3:
    st.header("How good are the summaries?")
    st.write("Compares our TextRank summarizer against two baselines using **ROUGE** "
             "(word overlap with human-written reference summaries).")
    src = st.radio("Reference data", ["Built-in demo (5 articles)", "Upload CSV with reference summaries"], horizontal=True)
    refs = None
    if src.startswith("Built-in"):
        refs = dl.load_reference_csv("data/sample_references.csv")
        st.info("Demo only: 5 articles is far too few for meaningful scores. Upload e.g. the CNN/DailyMail CSV "
                "(`article` + `highlights` columns) for a real evaluation.")
    else:
        up = st.file_uploader("CSV file", type="csv", key="ref_upload")
        if up is not None:
            raw = pd.read_csv(up)
            c1, c2 = st.columns(2)
            tcol = c1.selectbox("Article text column", raw.columns)
            rcol = c2.selectbox("Reference summary column", raw.columns, index=min(1, len(raw.columns) - 1))
            up.seek(0)
            try:
                refs = dl.load_reference_csv(up, tcol, rcol)
            except Exception as e:
                st.error(str(e))
    if refs is not None:
        c1, c2 = st.columns(2)
        n_eval = c1.slider("Sentences per summary", 1, 6, 3, key="n_eval")
        if len(refs) > 5:
            max_rows = c2.slider("Max articles to evaluate", 5, min(len(refs), 1000),
                                 min(len(refs), 200), key="max_rows")
        else:
            max_rows = len(refs)
            c2.caption(f"Using all {len(refs)} articles.")
        if st.button("Run evaluation", type="primary"):
            with st.spinner("Scoring…"):
                res = ev.evaluate(refs, n_sentences=n_eval, max_rows=max_rows)
            st.dataframe(res, use_container_width=True)
            plot = res[["ROUGE-1", "ROUGE-2", "ROUGE-L"]].reset_index().melt("System", var_name="Metric", value_name="F1")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.barplot(data=plot, x="Metric", y="F1", hue="System", ax=ax)
            ax.set_title(f"ROUGE F1 on {min(len(refs), max_rows)} articles")
            st.pyplot(fig)


# ---------- tab 4: insights ----------
with tab4:
    st.header("Dataset overview")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df, y="category", order=df["category"].value_counts().index, ax=ax, color="#4C78A8")
        ax.set_title("Articles per category"); ax.set_ylabel("")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=df, x="word_count", y="category", ax=ax, color="#72B7B2")
        ax.set_title("Article length by category"); ax.set_ylabel("")
        st.pyplot(fig)

    st.header("Classifier performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Hold-out accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Training articles", metrics["n_train"])
    m3.metric("Test articles", metrics["n_test"])

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics["confusion"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=metrics["labels"], yticklabels=metrics["labels"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    with c4:
        st.subheader("Top keywords per category")
        st.dataframe(clf.top_keywords_per_class(model, n=8), use_container_width=True)
        st.subheader("Precision / recall / F1")
        st.dataframe(metrics["report"].round(2), use_container_width=True)

    st.header("Model comparison (cross-validated)")
    comp = get_comparison(df)
    c5, c6 = st.columns([1, 1])
    c5.dataframe(comp, use_container_width=True)
    with c6:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=comp["Accuracy"], y=comp.index, xerr=comp["Accuracy std"], ax=ax, color="#4C78A8")
        ax.set_xlim(0, 1.05); ax.set_ylabel(""); ax.set_title("CV accuracy (± std)")
        st.pyplot(fig)

    st.header("Topic clustering (KMeans, unsupervised)")
    cl = get_clusters(df)
    st.metric("Adjusted Rand Index vs true categories", f"{cl['ari']:.2f}",
              help="1.0 = clusters perfectly match the real categories, 0 = random.")
    c7, c8 = st.columns(2)
    with c7:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=cl["coords"][:, 0], y=cl["coords"][:, 1], hue=df["category"], style=cl["labels"].astype(str),
                        ax=ax, s=60)
        ax.set_title("Articles in 2-D (colour = true category, marker = cluster)")
        ax.legend(fontsize=6, loc="best")
        st.pyplot(fig)
    with c8:
        st.subheader("Top terms per cluster")
        st.dataframe(cl["cluster_terms"], use_container_width=True)

    if len(df) < 200:
        st.info("The built-in dataset is tiny, so these numbers are only illustrative. "
                "Upload the BBC News CSV (`bbc-text.csv`, already has `category` and `text` columns) for realistic results.")