"""
Part-of-Speech (POS) Tagging Module.
Tags tokens to distinguish verbs (actions), nouns (objects/destinations), and adjectives (colors/attributes).
"""

from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class POSTagger:
    """NLTK-based Part-of-Speech tagger with deterministic rule-based fallback."""

    def __init__(self):
        self._has_nltk = False
        try:
            import nltk
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
                self._has_nltk = True
            except Exception:
                try:
                    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
                    self._has_nltk = True
                except Exception:
                    self._has_nltk = False
        except Exception:
            self._has_nltk = False

    def tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Produce POS tags for given tokens.
        Returns: List of (word, tag) tuples, e.g. [('pick', 'VB'), ('red', 'JJ'), ('bottle', 'NN')]
        """
        if not tokens:
            return []

        if self._has_nltk:
            try:
                from nltk import pos_tag
                return pos_tag(tokens)
            except Exception:
                pass

        # Robust heuristic robotic POS tagger
        tagged = []
        action_verbs = {"pick", "place", "move", "grasp", "lift", "put", "take", "bring", "drop", "release"}
        colors = {"red", "green", "blue", "yellow", "orange", "black", "white"}
        spatial_preps = {"in", "inside", "on", "into", "onto", "under", "next", "to", "at"}

        for t in tokens:
            t_lower = t.lower()
            if t_lower in action_verbs:
                tagged.append((t, "VB"))
            elif t_lower in colors:
                tagged.append((t, "JJ"))
            elif t_lower in spatial_preps:
                tagged.append((t, "IN"))
            elif t_lower in {"bottle", "box", "cup", "table", "can", "apple", "bin", "tray"}:
                tagged.append((t, "NN"))
            else:
                tagged.append((t, "NN"))
        return tagged
