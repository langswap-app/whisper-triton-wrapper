from pathlib import Path
from typing import Any, Dict, List, Tuple

import gc
import torch
import librosa
import numpy as np
import triton_python_backend_utils as pb_utils
from faster_whisper import WhisperModel

import json


THIS_DIR = Path(__file__).parent
MODEL_PATH = str(THIS_DIR / 'distil-whisper-large-v2/snapshots/fbac75ff0e24f79469c38f9c52517ae4c2b89198')

class TritonPythonModel:
    _whisper_model: WhisperModel
    _number_tokens: List[int]
    _align_models: Dict[str, Any]
    _device: str
    _output_type: Any

    def initialize(self, args: Dict[str, Any]):
        self._device = "cuda" #'cuda' if torch.cuda.is_available() else 'cpu'
        self._whisper_model = WhisperModel(MODEL_PATH, 
            device=self._device, 
            compute_type="float16"
        )
        self._number_tokens = [-1] + json.loads(
            (THIS_DIR / "number_tokens.json").read_text()
        )["number_tokens"]

    def extract_alignments(self, segments):
        word_segments = []
        for segment in segments:
            for word in segment.words:
                word_segments.append({'start': word.start, "end":word.end, 'word':word.word})
        return word_segments

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
                    audio_signal,
                    orig_sr = sampling_rate, 
                    target_sr = 16000
                )

            segments, info = self._whisper_model.transcribe(
                audio=audio_signal,
                suppress_tokens=self._number_tokens,
                word_timestamps=True
            )
            word_segments = self.extract_alignments(segments)
            language = info.language if language == "auto" else language

            result = {
                        "text": "".join([i['word'] for i in word_segments]),
                        "word_segments": word_segments
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
        return responses

    def finalize(self):
        del self._whisper_model
        torch.cuda.empty_cache()
        gc.collect()
