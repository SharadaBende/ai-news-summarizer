"""OPTIONAL abstractive summarizer using a Hugging Face transformer.

Needs:  pip install -r requirements-optional.txt   (transformers + torch, ~2 GB)
The model downloads on first use and needs internet. The rest of the project works without this.
"""
import importlib.util
from functools import lru_cache

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


def is_available() -> bool:
    return all(importlib.util.find_spec(p) is not None for p in ("transformers", "torch"))


@lru_cache(maxsize=2)
def _load(model_name: str):
    from transformers import pipeline
    return pipeline("summarization", model=model_name)


def summarize_abstractive(text: str, max_length: int = 120, min_length: int = 30,
                          model_name: str = DEFAULT_MODEL) -> str:
    if not is_available():
        raise RuntimeError("transformers/torch not installed. Run: pip install -r requirements-optional.txt")
    n_words = len(text.split())
    max_length = max(min_length + 5, min(max_length, int(n_words * 0.8)))
    out = _load(model_name)(text, max_length=max_length, min_length=min(min_length, max_length - 1),
                            do_sample=False, truncation=True)
    return out[0]["summary_text"]
