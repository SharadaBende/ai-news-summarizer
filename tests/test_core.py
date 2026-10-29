"""Run with:  python -m pytest -q   (pip install -r requirements-dev.txt)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import classifier as clf
import clustering
import data_loader as dl
import evaluation as ev
import summarizer as sm

TEXT = ("The central bank raised interest rates on Monday to fight inflation. Economists said the move would slow "
        "borrowing across the country. The governor warned that growth could weaken if prices keep climbing. "
        "Stock markets fell after the announcement as investors reassessed their forecasts. Analysts expect another "
        "decision next quarter.")


def test_clean_text_strips_html_and_urls():
    assert dl.clean_text("<p>Hello &amp; welcome http://x.com  now</p>") == "Hello & welcome now"


def test_split_sentences_ignores_short_fragments():
    assert len(sm.split_sentences("Hi. " + TEXT)) == 5


def test_summary_length_and_order():
    out = sm.summarize(TEXT, 2)
    sents = sm.split_sentences(TEXT)
    assert len(out) == 2
    assert [sents.index(s) for s in out] == sorted(sents.index(s) for s in out)


def test_short_text_returned_whole():
    assert len(sm.summarize("This is one reasonably long sentence here.", 3)) == 1


def test_keywords_non_empty():
    assert len(sm.extract_keywords(TEXT, 5)) == 5


def test_rouge_identical_and_disjoint():
    assert ev.rouge_scores("the cat sat on the mat", "the cat sat on the mat")["ROUGE-L"] == 1.0
    assert ev.rouge_n("alpha beta", "gamma delta", 1) == 0.0


def test_rouge_partial_overlap():
    assert 0 < ev.rouge_n("the cat sat", "the cat ran away", 1) < 1


def test_evaluate_returns_all_systems():
    refs = dl.load_reference_csv("data/sample_references.csv")
    res = ev.evaluate(refs, n_sentences=2)
    assert set(res.index) == set(ev.SYSTEMS)
    assert (res["ROUGE-1"] >= 0).all()


def test_classifier_beats_chance():
    df = dl.load_csv()
    _, m = clf.train(df)
    assert m["accuracy"] > 1 / df["category"].nunique()


def test_model_comparison_has_three_models():
    assert len(clf.compare_models(dl.load_csv())) == 3


def test_clustering_shapes():
    df = dl.load_csv()
    res = clustering.cluster_articles(df)
    assert len(res["labels"]) == len(df) and res["coords"].shape == (len(df), 2)
