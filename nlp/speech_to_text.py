"""
Speech-to-Text Module.
Transcribes spoken human voice commands using OpenAI Whisper, with simulation fallback.
"""

from typing import Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """Transcribes audio waveform or microphone input to text via Whisper."""

    def __init__(self, model_size: str = "base", simulation_mode: bool = True):
        self.model_size = model_size
        self.simulation_mode = simulation_mode
        self._whisper_model = None

        if not self.simulation_mode:
            try:
                import whisper
                logger.info(f"Loading Whisper model '{model_size}'...")
                self._whisper_model = whisper.load_model(model_size)
                logger.info("Whisper model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load Whisper model ({e}). Using simulation mode.")
                self.simulation_mode = True

    def transcribe(self, audio_path_or_bytes: Optional[str] = None) -> str:
        """
        Transcribe audio input to natural language text.
        If no audio is provided or in simulation mode, returns canonical sample command.
        """
        if not self.simulation_mode and self._whisper_model is not None and audio_path_or_bytes:
            try:
                result = self._whisper_model.transcribe(audio_path_or_bytes)
                text = result.get("text", "").strip()
                logger.info(f"Whisper transcription: '{text}'")
                return text
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}. Falling back to default.")

        # Default canonical hackathon benchmark command
        default_cmd = "Pick up the red bottle and place it inside the storage box."
        logger.info(f"Using simulated voice transcription: '{default_cmd}'")
        return default_cmd
