"""
Tokenization Module.
Performs sentence and word-level tokenization for robotic voice commands.
"""

from typing import List
import re


class Tokenizer:
    """Sentence and word tokenizer with robust regex and NLTK fallback."""

    def __init__(self):
        self._has_nltk = False
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
                self._has_nltk = True
            except Exception:
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                    self._has_nltk = True
                except Exception:
                    self._has_nltk = False
        except Exception:
            self._has_nltk = False

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        if self._has_nltk:
            try:
                from nltk.tokenize import sent_tokenize
                return sent_tokenize(text)
            except Exception:
                pass
        # Regex fallback
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """Split text into word tokens preserving punctuation-free words."""
        if not text:
            return []
        if self._has_nltk:
            try:
                from nltk.tokenize import word_tokenize
                return word_tokenize(text)
            except Exception:
                pass
        # Regex fallback
        return re.findall(r'\b\w+\b', text)
