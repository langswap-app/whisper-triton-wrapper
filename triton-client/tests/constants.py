from pathlib import Path

THIS_DIR = Path(__file__).parent

DEFAULT_TTS_TEXT = """
The sun was setting on the horizon, casting a warm glow over the peaceful countryside.
Birds chirped in the distance, their songs harmonizing with the rustling of leaves in the gentle breeze.
A lone traveler walked along the dirt path, his footsteps crunching against the rocks beneath his feet. Wow!
"""
DEFAULT_ASR_AUDIO = THIS_DIR / "sample_asr_audio.wav"
DEFAULT_ASR_LANGUAGE = "en"
