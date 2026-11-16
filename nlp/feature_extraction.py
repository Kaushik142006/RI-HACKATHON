"""
Feature Extraction Module.
Extracts TF-IDF n-gram features from preprocessed robotic commands.
"""

from typing import List, Union
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class TfidfFeatureExtractor:
    """Computes TF-IDF vector representations with unigrams and bigrams."""

    def __init__(self, ngram_range=(1, 2), max_features: int = 500):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "TfidfFeatureExtractor":
        """Fit vectorizer on a corpus of preprocessed commands."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into TF-IDF sparse matrix/array."""
        if not self.is_fitted:
            raise RuntimeError("TfidfFeatureExtractor must be fitted before transform.")
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        res = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return res.toarray()
