"""
Command Parser & Structured Command Output Module.
Combines Preprocessing, Intent Classification, and Entity Extraction into a validated Pydantic model.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from nlp.preprocessing import CommandPreprocessor
from nlp.intent_classifier import IntentClassifier
from nlp.entity_extraction import EntityExtractor


class StructuredCommand(BaseModel):
    raw_command: str
    intent: str
    intent_confidence: float
    action: str
    target_object: Optional[str] = None
    target_color: Optional[str] = None
    destination: Optional[str] = None
    spatial_preposition: Optional[str] = None
    tokens: List[str] = Field(default_factory=list)
    lemmas: List[str] = Field(default_factory=list)
    is_valid: bool = True
    validation_error: Optional[str] = None


class CommandParser:
    """End-to-end natural language command parser for robotic manipulation."""

    def __init__(self, classifier_type: str = "svm"):
        self.preprocessor = CommandPreprocessor()
        self.classifier = IntentClassifier(model_type=classifier_type)
        self.entity_extractor = EntityExtractor()

    def parse(self, text: str) -> StructuredCommand:
        """Parse natural language command into structured schema."""
        prep_result = self.preprocessor.process(text)
        intent, confidence = self.classifier.predict(prep_result["processed_string"])
        entities = self.entity_extractor.extract(text)

        # Validation
        is_valid = True
        val_error = None
        if not entities["target_object"] and intent == "pick_and_place":
            is_valid = False
            val_error = "Missing target object in pick_and_place command."

        return StructuredCommand(
            raw_command=text,
            intent=intent,
            intent_confidence=confidence,
            action=entities["action"],
            target_object=entities["target_object"],
            target_color=entities["target_color"],
            destination=entities["destination"],
            spatial_preposition=entities["spatial_preposition"],
            tokens=prep_result["tokens"],
            lemmas=prep_result["lemmas"],
            is_valid=is_valid,
            validation_error=val_error
        )
