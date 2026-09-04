"""Text normalization and TF-IDF feature construction."""
import re
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """Normalize resume text while retaining useful technical tokens such as C++ and node.js."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vectorizer() -> TfidfVectorizer:
    """Create the shared vectorizer used by all personality regressors."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=2500,
        sublinear_tf=True,
    )


def fit_transform_texts(texts: Iterable[str]):
    vectorizer = build_vectorizer()
    return vectorizer, vectorizer.fit_transform(texts)

