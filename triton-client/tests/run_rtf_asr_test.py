import argparse
import time
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from constants import DEFAULT_ASR_AUDIO, DEFAULT_ASR_LANGUAGE
from tqdm import tqdm

from tts_client.asr_client import TritonASRClient
from tts_client.constants import DEFAULT_ASR_GRPC_API_URL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        "-u",
        help="URL of the Triton Inference Server GRPC API.",
        type=str,
        default=DEFAULT_ASR_GRPC_API_URL,
    )
    parser.add_argument(
        "--audio",
        "-a",
        help="Audio to transcribe",
        type=Path,
        default=DEFAULT_ASR_AUDIO,
    )
    parser.add_argument(
        "--language",
        "-l",
        help="Language of audio",
        type=str,
        default=DEFAULT_ASR_LANGUAGE,
    )
    parser.add_argument(
        "--num_runs",
        "-n",
        help="Number of runs to measure RTF",
        type=int,
        default=100,
    )
    return parser.parse_args()


def test_rtf(
    url: str, audio: Path, language: str, num_runs: int
) -> Tuple[float, float]:
    """
    Measure RTF of the ASR model
    """
    client = TritonASRClient(url=url)
    audio, sampling_rate = librosa.load(audio, sr=16000)
    audio_duration = len(audio) / sampling_rate

    rtf_list = []
    for _ in tqdm(range(num_runs), desc="Measuring RTF"):
        start_time = time.time()
        _ = client.transcribe(
            audio_signal=audio, sampling_rate=sampling_rate, language=language
        )
        rtf_list.append((time.time() - start_time) / audio_duration)

    return float(np.mean(rtf_list)), float(np.std(rtf_list))


def main():
    args = parse_args()
    mean_rtf, std_rtf = test_rtf(
        url=args.url,
        audio=args.audio,
        language=args.language,
        num_runs=args.num_runs,
    )
    print(f"Mean RTF (lower is better): {mean_rtf:.3f} +- {std_rtf:.4f}")


if __name__ == "__main__":
    main()
