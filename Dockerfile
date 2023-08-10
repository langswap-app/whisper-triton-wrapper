FROM nvcr.io/nvidia/tritonserver:22.04-pyt-python-py3

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    espeak-ng \
 && rm -rf /var/lib/apt/lists/*

ADD requirements.txt .

RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

COPY model_repository/ model_repository/

RUN python3 download.py

CMD tritonserver --grpc-port=8085 --http-port=8005 --model-repository=model_repository/ --metrics-port=60189 --log-info=true --model-control-mode=explicit
