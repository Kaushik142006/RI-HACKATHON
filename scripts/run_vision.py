"""
Script: Run Vision Pipeline.
Captures an RGB-D frame, runs detection, segmentation, tracking, color identification,
depth extraction, 3D localization, and prints the unified structured vision state.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.vision_pipeline import VisionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_vision")


def main():
    logger.info("Initializing Computer Vision Pipeline...")
    pipeline = VisionPipeline(simulation_mode=True)
    pipeline.start()

    logger.info("Executing single perception cycle...")
    state = pipeline.process_frame()

    print("\n" + "="*50)
    print("UNIFIED STRUCTURED VISION STATE OUTPUT:")
    print("="*50)
    print(json.dumps(state.model_dump(), indent=2))
    print("="*50 + "\n")

    pipeline.stop()
    logger.info("Vision Pipeline run complete.")


if __name__ == "__main__":
    main()
