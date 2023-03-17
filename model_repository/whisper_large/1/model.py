from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import triton_python_backend_utils as pb_utils
import whisper


THIS_DIR = Path(__file__).parent


class TritonPythonModel:
    _whisper_model: whisper.Whisper
    _output_type: Any

    def initialize(self, args: Dict[str, Any]):
        self._whisper_model = whisper.load_model(
            str(THIS_DIR / "large-v2.pt"), device="cuda:0"
        )

    def execute(self, requests: List[Any]):
        responses = []

        for request in requests:
            audio_signal = pb_utils.get_input_tensor_by_name(
                request,
                "audio_signal",
            ).as_numpy()

            sampling_rate = pb_utils.get_input_tensor_by_name(
                request,
                "sampling_rate",
            ).as_numpy()

            language = pb_utils.get_input_tensor_by_name(
                request,
                "language",
            ).as_numpy()
            language = language.astype('U')[0]

            if sampling_rate != 16000:
                audio_signal = librosa.resample(
                    audio_signal, sampling_rate, 16000
                )

            text_result = self._whisper_model.transcribe(
                audio=audio_signal,
                language=language,
            )["text"]

            text_output = pb_utils.Tensor(
                "transcription",
                np.array(
                    [text_result.encode("utf-8")], dtype=np.object_
                )
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[text_output]
            )
            responses.append(inference_response)

        return responses
