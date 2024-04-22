from typing import Tuple

import numpy as np
import tritonclient.grpc as grpcclient

from tts_client.base import BaseTritonClient
from tts_client.constants import (DEFAULT_TTS_GRPC_API_URL, SE_MODEL_NAME,
                                  SPEAKER_EMBEDDING_SIZE, TTS_MODEL_NAME)


class TritonTTSClient(BaseTritonClient):
    """
    TritonTTSClient is a client for the Triton Inference Server.

    Note: Works only with GRPC protocol (because it's faster, than HTTP)

    Args:
        url (str): URL of the Triton Inference Server GRPC API.
    """

    def __init__(self, url: str = DEFAULT_TTS_GRPC_API_URL):
        super().__init__(url=url, model_name=TTS_MODEL_NAME)

        # Load speaker encoder model. It is CPU model
        self.load_model(model_name=SE_MODEL_NAME)

    @property
    def sampling_rate(self) -> int:
        """Get sampling rate of the TTS model"""
        return 24000

    def synthesize(
        self,
        text: str,
        speaker_embedding: np.ndarray,
        length_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesize speech from text and speaker embedding

        Args:
            text (str): Text to synthesize
            speaker_embedding (np.ndarray): Speaker embedding to use for synthesis (FP32)
            length_scale (float): Length scale of the output audio

        Returns:
            1. Audio waveform, np.ndarray of type INT16 [-32768, 32768] with shape (audio_length,)
            2. Duration of each text_inputs token in spectrogram frames with shape (num_tokens,)
            3. Text inputs, np.ndarray of type INT32 with shape (num_tokens,)
            4. Word timestamps, np.ndarray of type INT32 with shape (num_tokens, 2) in milliseconds
        """
        string_as_array = np.array([text.encode("utf-8")], dtype=np.object_)
        inputs = [
            grpcclient.InferInput("input__0", [1], "BYTES"),
            grpcclient.InferInput("input__1", [SPEAKER_EMBEDDING_SIZE], "FP32"),
            grpcclient.InferInput("input__2", [1], "FP32"),
        ]
        inputs[0].set_data_from_numpy(string_as_array)
        inputs[1].set_data_from_numpy(speaker_embedding)
        inputs[2].set_data_from_numpy(
            np.array([length_scale]).astype(np.float32)
        )

        outputs = [
            grpcclient.InferRequestedOutput("output__0"),
            grpcclient.InferRequestedOutput("output__1"),
            grpcclient.InferRequestedOutput("output__2"),
            grpcclient.InferRequestedOutput("output__3"),
        ]

        results = self._client.infer(
            model_name=TTS_MODEL_NAME, inputs=inputs, outputs=outputs
        )
        audio = results.as_numpy("output__0")
        durations = results.as_numpy("output__1")
        text_inputs = results.as_numpy("output__2")
        word_positions = results.as_numpy("output__3")
        word_positions = [
            (word_positions[i], word_positions[i + 1])
            for i in range(0, len(word_positions), 2)
        ]
        return audio, durations, text_inputs, word_positions

    def get_speaker_embedding(
        self, audio: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """
        Get speaker embedding from audio

        Args:
            audio (np.ndarray): Audio waveform, np.ndarray of type FP32 [0, 1] with shape (audio_length,)
            sampling_rate (int): Sampling rate of audio

        Returns: Speaker embedding (FP32), np.ndarray of shape (SPEAKER_EMBEDDING_SIZE,)
        """
        inputs = [
            grpcclient.InferInput("input__0", [audio.shape[0]], "FP32"),
            grpcclient.InferInput("input__1", [1], "INT32"),
        ]
        inputs[0].set_data_from_numpy(audio)
        inputs[1].set_data_from_numpy(
            np.array([sampling_rate]).astype(np.int32)
        )

        outputs = [grpcclient.InferRequestedOutput("output__0")]
        results = self._client.infer(
            model_name=SE_MODEL_NAME, inputs=inputs, outputs=outputs
        )
        return results.as_numpy("output__0")
