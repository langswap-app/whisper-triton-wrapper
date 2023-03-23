FROM nvcr.io/nvidia/tritonserver:22.04-pyt-python-py3

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    espeak-ng \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt && pip install git+https://github.com/m-bain/whisperX.git

COPY . /app
WORKDIR /app

COPY model_repository/ model_repository/

RUN python3 download.py

CMD tritonserver --grpc-port=8085 --http-port=8005 --model-repository=model_repository/ --metrics-port=60189 --log-info=true
