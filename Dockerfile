FROM nvcr.io/nvidia/tritonserver:23.02-pyt-python-py3
COPY . /WHISPER
WORKDIR /WHISPER
RUN apt-get update || true
RUN apt-get install libsndfile1 ffmpeg -y
RUN pip install openai-whisper && pip install -r requirements.txt && pip install git+https://github.com/m-bain/whisperX.git
COPY model_repository/ model_repository/
CMD tritonserver --grpc-port=8085 --http-port=8005 --model-repository=model_repository/ --metrics-port=60189 --log-info=true