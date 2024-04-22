import json
from typing import Dict, List, Tuple, Union

import numpy as np
import tritonclient.grpc as grpcclient

from tts_client.base import BaseTritonClient
from tts_client.constants import (DEFAULT_ASR_GRPC_API_URL,
                                  WHISPER_LARGE_MODEL_NAME)


class TritonASRClient(BaseTritonClient):
    def __init__(self, url: str = DEFAULT_ASR_GRPC_API_URL):
        super().__init__(url=url, model_name=WHISPER_LARGE_MODEL_NAME)

    def transcribe(
        self, audio_signal: np.ndarray, sampling_rate: float, language: str
    ) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
        """
        Transcribe audio signal into text

        Args:
            audio_signal (np.ndarray): Audio signal as INT32 numpy array
            sampling_rate (float): Sampling rate of the audio signal
            language (str): Language of the audio signal

        Returns:
            transcription (str): Transcribed text
        """
        language_as_array = np.array(
            [language.encode("utf-8")], dtype=np.object_
        )
        inputs = [
            grpcclient.InferInput(
                "audio_signal", [audio_signal.shape[0]], "FP32"
            ),
            grpcclient.InferInput("sampling_rate", [1], "FP32"),
            grpcclient.InferInput("language", [1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(audio_signal.astype(np.float32))
        inputs[1].set_data_from_numpy(
            np.array([sampling_rate]).astype(np.float32)
        )
        inputs[2].set_data_from_numpy(language_as_array)

        outputs = [grpcclient.InferRequestedOutput("transcription")]

        results = self._client.infer(
            model_name=WHISPER_LARGE_MODEL_NAME, inputs=inputs, outputs=outputs
        )
        transcription = results.as_numpy("transcription")[0].decode("utf-8")
        transcription = json.loads(transcription)
        return transcription["text"], transcription["word_segments"]
