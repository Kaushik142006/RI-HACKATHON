"""
Stemming Module.
Implements PorterStemmer and SnowballStemmer for robotic command vocabulary normalization.
"""

from typing import List


class CommandStemmer:
    """Provides word stemming using Porter and Snowball stemmers."""

    def __init__(self, algorithm: str = "porter"):
        self.algorithm = algorithm.lower()
        self._porter = None
        self._snowball = None

        try:
            from nltk.stem import PorterStemmer, SnowballStemmer
            self._porter = PorterStemmer()
            self._snowball = SnowballStemmer(language="english")
            self._has_nltk = True
        except ImportError:
            self._has_nltk = False

    def stem_word(self, word: str) -> str:
        """Stem a single word."""
        if not self._has_nltk:
            # Basic fallback suffix stripper
            for suffix in ("ing", "ly", "ed", "es", "s"):
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    return word[:-len(suffix)]
            return word

        if self.algorithm == "snowball" and self._snowball:
            return self._snowball.stem(word)
        return self._porter.stem(word)

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem a list of tokens."""
        return [self.stem_word(t) for t in tokens]
