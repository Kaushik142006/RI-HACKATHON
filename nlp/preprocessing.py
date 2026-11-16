"""
NLP Preprocessing Pipeline.
Performs tokenization, lowercasing, selective stop-word removal (preserving critical spatial prepositions),
POS tagging, lemmatization, and stemming.
"""

from typing import List, Dict, Any, Tuple
import logging
from nlp.tokenization import Tokenizer
from nlp.stemming import CommandStemmer
from nlp.lemmatization import CommandLemmatizer
from nlp.pos_tagging import POSTagger

logger = logging.getLogger(__name__)

# Spatial prepositions and action particles that MUST be preserved for robotic commands
CRITICAL_SPATIAL_WORDS = {
    "in", "inside", "on", "into", "onto", "under", "below", "above",
    "next", "to", "beside", "from", "at", "left", "right", "front", "back", "up", "down"
}


class CommandPreprocessor:
    """End-to-end classical NLP text preprocessor."""

    def __init__(self, stem_algorithm: str = "porter"):
        self.tokenizer = Tokenizer()
        self.pos_tagger = POSTagger()
        self.lemmatizer = CommandLemmatizer()
        self.stemmer = CommandStemmer(algorithm=stem_algorithm)

        # Standard stop words minus spatial prepositions
        self._standard_stopwords = {
            "a", "an", "the", "and", "or", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "please", "could", "would", "you", "me",
            "it", "this", "that", "these", "those"
        }

    def clean_text(self, text: str) -> str:
        """Lowercase and normalize whitespace."""
        if not text:
            return ""
        return " ".join(text.lower().strip().split())

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Selectively remove useless filler stop-words while strictly preserving
        spatial relations ('in', 'on', 'inside', 'under', 'to', etc.).
        """
        filtered = []
        for t in tokens:
            t_lower = t.lower()
            if t_lower in CRITICAL_SPATIAL_WORDS:
                filtered.append(t_lower)
            elif t_lower not in self._standard_stopwords:
                filtered.append(t_lower)
        return filtered

    def process(self, text: str) -> Dict[str, Any]:
        """
        Execute full preprocessing pipeline.
        Returns:
            Dict containing raw_text, clean_text, tokens, filtered_tokens,
            pos_tags, lemmas, stems.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenizer.tokenize_words(cleaned)
        filtered_tokens = self.filter_stopwords(tokens)

        pos_tags = self.pos_tagger.tag(filtered_tokens)
        lemmas = self.lemmatizer.lemmatize_tagged_tokens(pos_tags)
        stems = self.stemmer.stem_tokens(lemmas)

        return {
            "raw_text": text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "pos_tags": pos_tags,
            "lemmas": lemmas,
            "stems": stems,
            "processed_string": " ".join(lemmas)
        }
