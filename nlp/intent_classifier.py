"""
Intent Classification Module.
Classifies robotic commands using Scikit-learn SVM with Logistic Regression alternative.
"""

from typing import Tuple, Dict, Any, List
import logging
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from nlp.feature_extraction import TfidfFeatureExtractor

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies user intent (e.g. pick_and_place, inspect, stop) from command text."""

    DEFAULT_TRAINING_DATA = [
        ("pick up the red bottle and place it inside the storage box", "pick_and_place"),
        ("grab the blue cup and put it on the tray", "pick_and_place"),
        ("lift the green apple and drop it into the basket", "pick_and_place"),
        ("take the yellow can and place it in the bin", "pick_and_place"),
        ("pick the bottle and put it in the box", "pick_and_place"),
        ("place the scissors inside the container", "pick_and_place"),
        ("move the red cup to the table corner", "pick_and_place"),
        ("inspect the workspace", "inspect"),
        ("check if the table is clean", "inspect"),
        ("scan the environment for obstacles", "inspect"),
        ("look at the objects on the table", "inspect"),
        ("stop the robot immediately", "emergency_stop"),
        ("halt execution", "emergency_stop"),
        ("pause movement", "emergency_stop"),
        ("clear the table of clutter", "clean_workspace"),
        ("tidy up the workspace", "clean_workspace"),
        ("where is the red bottle", "query_location"),
        ("what do you see on the table", "query_scene")
    ]

    def __init__(self, model_type: str = "svm"):
        self.model_type = model_type.lower()
        self.feature_extractor = TfidfFeatureExtractor()

        if self.model_type == "logistic_regression":
            self.classifier = LogisticRegression(class_weight="balanced", random_state=42)
        else:
            self.classifier = SVC(probability=True, kernel="linear", random_state=42)

        self.is_trained = False
        # Train immediately on baseline robotic dataset
        self.train_default()

    def train_default(self) -> None:
        """Train classifier on embedded standard dataset."""
        texts = [item[0] for item in self.DEFAULT_TRAINING_DATA]
        labels = [item[1] for item in self.DEFAULT_TRAINING_DATA]
        self.train(texts, labels)

    def train(self, texts: List[str], labels: List[str]) -> None:
        """Train classifier on provided texts and labels."""
        X = self.feature_extractor.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        logger.info(f"Trained {self.model_type.upper()} intent classifier on {len(texts)} samples.")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict command intent and associated confidence.
        Returns:
            Tuple of (intent: str, confidence: float)
        """
        if not self.is_trained:
            self.train_default()

        X = self.feature_extractor.transform([text])
        predicted_intent = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        return str(predicted_intent), round(confidence, 3)
