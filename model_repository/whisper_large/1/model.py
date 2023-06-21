from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import triton_python_backend_utils as pb_utils
import whisperx

import json


THIS_DIR = Path(__file__).parent


class TritonPythonModel:
    _whisper_model: whisperx.Whisper
    _align_models: Dict[str, Any]
    _device: str
    _output_type: Any

    def initialize(self, args: Dict[str, Any]):
        self._device = "cuda:0"
        self._whisper_model = whisperx.load_model(
            str(THIS_DIR / "large-v2.pt"), device=self._device
        )

        self._align_models = {}

    def load_align_model(self, language: str) -> Tuple[Any, Any]:
        return whisperx.load_align_model(
                language_code=language, device=self._device
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

            result = self._whisper_model.transcribe(
                audio=audio_signal,
            )

            language = result["language"] if language == "auto" else language

            # Initialize align model
            model_a, metadata = self.load_align_model(language)

            result_aligned = whisperx.align(
                result["segments"], model_a, metadata, audio_signal, self._device
            )

            result = {
                "text": " ".join([i["text"] for i in result_aligned["segments"]]),
                "word_segments": result_aligned["word_segments"]
            }

            text_output = pb_utils.Tensor(
                "transcription",
                np.array(
                    [json.dumps(result).encode("utf-8")], dtype=np.object_
                )
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[text_output]
            )
            responses.append(inference_response)

            # TODO (a.gribul): Remove this dirty hack
            del model_a
            
        return responses
