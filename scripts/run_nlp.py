"""
Script: Run Natural Language Processing Pipeline.
Transcribes voice input, preprocesses command with NLTK, classifies intent,
and extracts target object, color, destination, and spatial prepositions.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.speech_to_text import SpeechTranscriber
from nlp.command_parser import CommandParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_nlp")


def main():
    logger.info("Initializing NLP Pipeline...")
    stt = SpeechTranscriber(simulation_mode=True)
    parser = CommandParser(classifier_type="svm")

    # Step 1: Transcribe voice
    text = stt.transcribe()
    logger.info(f"Transcribed Text: '{text}'")

    # Step 2: Parse structured command
    parsed = parser.parse(text)

    print("\n" + "="*50)
    print("STRUCTURED NLP COMMAND OUTPUT:")
    print("="*50)
    print(json.dumps(parsed.model_dump(), indent=2))
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
