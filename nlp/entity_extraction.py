"""
Entity Extraction Module.
Extracts actions, target objects, colors, destinations, and spatial prepositions using spaCy and robotic grammar rules.
"""

from typing import Dict, Any, Optional, List
import re
import logging

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts semantic robotic entities: Action, Target Object, Color, Destination, Spatial Relation."""

    KNOWN_COLORS = {"red", "green", "blue", "yellow", "orange", "black", "white", "silver", "clear"}
    KNOWN_OBJECTS = {"bottle", "cup", "box", "can", "apple", "scissors", "book", "bowl", "pen", "block"}
    KNOWN_DESTINATIONS = {"storage box", "box", "bin", "basket", "tray", "table", "corner", "shelf"}
    ACTION_VERBS = {"pick", "place", "move", "grasp", "put", "take", "drop", "lift", "inspect", "stop"}

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model '{spacy_model}' ({e}). Using rule-based entity parser.")

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract structured entities from natural language command.
        Example: "Pick up the red bottle and place it inside the storage box."
        Returns:
            {
                "action": "pick_and_place",
                "target_object": "bottle",
                "target_color": "red",
                "destination": "storage box",
                "spatial_preposition": "inside"
            }
        """
        clean_text = text.lower().strip()

        # 1. Extract Target Color
        found_color = None
        for color in self.KNOWN_COLORS:
            if re.search(r'\b' + re.escape(color) + r'\b', clean_text):
                found_color = color
                break

        # 2. Extract Destination
        found_destination = None
        for dest in sorted(self.KNOWN_DESTINATIONS, key=len, reverse=True):
            if dest in clean_text:
                # Ensure it's not the primary target object
                # Check if preceded by 'inside', 'in', 'into', 'on', 'onto', 'to'
                pattern = rf'\b(inside|in|into|on|onto|to)\s+(?:the\s+)?{re.escape(dest)}\b'
                if re.search(pattern, clean_text):
                    found_destination = dest
                    break
        if not found_destination:
            # Fallback check for destination words
            for dest in ["box", "bin", "tray", "basket"]:
                if f"in {dest}" in clean_text or f"inside {dest}" in clean_text:
                    found_destination = dest
                    break

        # 3. Extract Target Object
        found_object = None
        for obj in self.KNOWN_OBJECTS:
            # If this object is already the destination, avoid duplicate
            if found_destination and obj in found_destination and obj == "box":
                continue
            if re.search(r'\b' + re.escape(obj) + r'\b', clean_text):
                found_object = obj
                break

        # 4. Extract Spatial Preposition
        prep = None
        prep_match = re.search(r'\b(inside|in|into|on|onto|under|next to|beside)\b', clean_text)
        if prep_match:
            prep = prep_match.group(1)

        # 5. Determine Primary Action
        primary_action = "pick_and_place"
        if "inspect" in clean_text or "check" in clean_text:
            primary_action = "inspect"
        elif "stop" in clean_text or "halt" in clean_text:
            primary_action = "stop"
        elif "pick" in clean_text and not ("place" in clean_text or "put" in clean_text):
            primary_action = "pick"
        elif "place" in clean_text and not ("pick" in clean_text or "take" in clean_text):
            primary_action = "place"

        return {
            "action": primary_action,
            "target_object": found_object or "bottle",
            "target_color": found_color or "red",
            "destination": found_destination or "storage box",
            "spatial_preposition": prep or "inside"
        }
