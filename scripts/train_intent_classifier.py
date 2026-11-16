"""
Script: Train Intent Classifier.
Loads training data from CSV, trains SVM / Logistic Regression classifier, and evaluates performance.
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.intent_classifier import IntentClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_nlp")


def main():
    logger.info("Training NLP Intent Classifier on robotic command dataset...")
    data_path = Path(__file__).resolve().parent.parent / "nlp" / "dataset" / "robotic_commands.csv"

    texts = []
    labels = []

    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cmd = row.get("command", "").strip()
                intent = row.get("intent", "").strip()
                if cmd and intent:
                    texts.append(cmd)
                    labels.append(intent)

    classifier = IntentClassifier(model_type="svm")
    if texts:
        classifier.train(texts, labels)
        logger.info(f"Trained SVM classifier on {len(texts)} custom dataset samples.")

    # Benchmark test evaluation
    test_cases = [
        "Pick up the red bottle and place it inside the storage box.",
        "Inspect the table workspace.",
        "Stop robot movement immediately."
    ]

    for t in test_cases:
        intent, conf = classifier.predict(t)
        logger.info(f"Input: '{t}' -> Predicted Intent: '{intent}' (Confidence: {conf:.2f})")


if __name__ == "__main__":
    main()
