"""
Unit Tests for NLP Pipeline.
Tests tokenization, stemming, lemmatization, POS tagging, selective stop-word retention,
SVM/Logistic Regression intent classification, and entity extraction.
"""

import pytest
from nlp.preprocessing import CommandPreprocessor, CRITICAL_SPATIAL_WORDS
from nlp.tokenization import Tokenizer
from nlp.stemming import CommandStemmer
from nlp.lemmatization import CommandLemmatizer
from nlp.pos_tagging import POSTagger
from nlp.intent_classifier import IntentClassifier
from nlp.entity_extraction import EntityExtractor
from nlp.command_parser import CommandParser


def test_spatial_preposition_retention():
    preprocessor = CommandPreprocessor()
    text = "Pick up the red bottle and place it inside the storage box."
    res = preprocessor.process(text)
    filtered = res["filtered_tokens"]
    # Verify spatial word 'inside' is preserved
    assert "inside" in filtered
    # Verify non-informative word 'the' was removed
    assert "the" not in filtered


def test_stemming_and_lemmatization():
    stemmer = CommandStemmer()
    assert stemmer.stem_word("picking") == "pick"

    lemmatizer = CommandLemmatizer()
    assert lemmatizer.lemmatize_word("bottles", "N") == "bottle"


def test_intent_classifier():
    classifier = IntentClassifier(model_type="svm")
    intent, conf = classifier.predict("Pick up the red bottle and place it inside the storage box")
    assert intent == "pick_and_place"
    assert conf > 0.5


def test_entity_extraction():
    extractor = EntityExtractor()
    text = "Pick up the red bottle and place it inside the storage box."
    entities = extractor.extract(text)
    assert entities["target_object"] == "bottle"
    assert entities["target_color"] == "red"
    assert "box" in entities["destination"]
    assert entities["spatial_preposition"] == "inside"


def test_command_parser_full():
    parser = CommandParser(classifier_type="svm")
    res = parser.parse("Pick up the red bottle and place it inside the storage box.")
    assert res.is_valid
    assert res.target_object == "bottle"
    assert res.target_color == "red"
