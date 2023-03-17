## OpenAI Whisper (large)

Triton serving for OpenAI Whisper (large-v2).

### Usage

Download model:

```bash
python download.py
```

Build docker image:

```bash
sudo docker build -t tts_whisper_v1 
```

Run docker image:

```bash
sudo docker run --network host --gpus all asr_whisper_v1:latest
```

GRPC PORT: 8085 ; HTTP PORT: 8005