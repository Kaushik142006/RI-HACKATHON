"""
Lemmatization Module.
Implements WordNetLemmatizer with automatic POS mapping and robust fallback.
"""

from typing import List, Tuple


class CommandLemmatizer:
    """Lemmatizes action verbs and object nouns to canonical dictionary roots."""

    def __init__(self):
        self._lemmatizer = None
        self._has_nltk = False
        try:
            import nltk
            try:
                nltk.data.find('corpora/wordnet')
                from nltk.stem import WordNetLemmatizer
                self._lemmatizer = WordNetLemmatizer()
                self._has_nltk = True
            except Exception:
                self._has_nltk = False
        except Exception:
            self._has_nltk = False

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Map POS tag to first character lemmatize() accepts."""
        if treebank_tag.startswith('J'):
            return 'a'  # Adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # Verb
        elif treebank_tag.startswith('N'):
            return 'n'  # Noun
        elif treebank_tag.startswith('R'):
            return 'r'  # Adverb
        else:
            return 'n'

    def lemmatize_word(self, word: str, pos_tag: str = 'n') -> str:
        """Lemmatize a word with given POS tag."""
        if not self._has_nltk or not self._lemmatizer:
            w = word.lower()
            if w.endswith("ies") and len(w) > 4:
                return w[:-3] + "y"
            if w.endswith("es") and len(w) > 4:
                return w[:-2]
            if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
                return w[:-1]
            return w

        wn_pos = self._get_wordnet_pos(pos_tag)
        try:
            return self._lemmatizer.lemmatize(word.lower(), pos=wn_pos)
        except Exception:
            return word.lower()

    def lemmatize_tagged_tokens(self, tagged_tokens: List[Tuple[str, str]]) -> List[str]:
        """Lemmatize a sequence of (word, pos_tag) pairs."""
        return [self.lemmatize_word(word, tag) for word, tag in tagged_tokens]
