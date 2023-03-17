from pathlib import Path
import logging
import subprocess
logger = logging.getLogger(__name__)

DOWNLOAD_URL = "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
DOWNLOAD_PATH = Path(__file__).parent / "model_repository" / "whisper_large" / "1" / "large-v2.pt"

logger.info(f"Downloading model from {DOWNLOAD_URL} to {DOWNLOAD_PATH}...")
subprocess.call(["wget", DOWNLOAD_URL, "-O", str(DOWNLOAD_PATH)])
logger.info("Download complete.")

