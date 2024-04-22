import argparse
import time
from typing import Tuple

import numpy as np
from constants import DEFAULT_TTS_TEXT
from tqdm import tqdm

from tts_client import TritonTTSClient
from tts_client.constants import DEFAULT_TTS_GRPC_API_URL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        "-u",
        help="URL of the Triton Inference Server GRPC API.",
        type=str,
        default=DEFAULT_TTS_GRPC_API_URL,
    )
    parser.add_argument(
        "--text",
        "-t",
        help="Text to synthesize",
        type=str,
        default=DEFAULT_TTS_TEXT,
    )
    parser.add_argument(
        "--num_runs",
        "-n",
        help="Number of runs to measure RTF",
        type=int,
        default=100,
    )
    return parser.parse_args()


def test_rtf(url: str, text: str, num_runs: int) -> Tuple[float, float]:
    """
    Measure RTF of the TTS model
    """
    client = TritonTTSClient(url=url)
    speaker_embedding = np.random.rand(512).astype(np.float32)

    rtf_list = []
    for _ in tqdm(range(num_runs), desc="Measuring RTF"):
        start_time = time.time()
        audio = client.synthesize(
            text=text, speaker_embedding=speaker_embedding
        )
        rtf_list.append(
            (time.time() - start_time) / (len(audio) / client.sampling_rate)
        )

    return float(np.mean(rtf_list)), float(np.std(rtf_list))


def main():
    args = parse_args()
    mean_rtf, std_rtf = test_rtf(
        url=args.url, text=args.text, num_runs=args.num_runs
    )
    print(f"Mean RTF (lower is better): {mean_rtf:.3f} +- {std_rtf:.4f}")


if __name__ == "__main__":
    main()
