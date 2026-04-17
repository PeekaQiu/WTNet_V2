FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10.20

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libnss3-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    git \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    liblzma-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf Python-${PYTHON_VERSION}*

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        torch==2.10.0 \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

RUN pip3 install --no-cache-dir \
        addict==2.4.0 \
        einops==0.8.2 \
        future==1.0.0 \
        gdown==6.0.0 \
        h5py==3.16.0 \
        joblib==1.5.3 \
        lmdb==2.2.0 \
        lpips==0.1.4 \
        matplotlib==3.10.8 \
        natsort==8.4.0 \
        numpy \
        opencv-python==4.13.0.92 \
        pillow \
        PyWavelets==1.8.0 \
        PyYAML==6.0.3 \
        requests==2.33.1 \
        scikit-image==0.25.2 \
        scikit-learn==1.7.2 \
        scipy==1.15.3 \
        tqdm==4.67.3 \
        yacs==0.1.8 \
        yapf==0.43.0 \
	tensorboard==2.20.0 \
	absl-py==2.4.0 \
        grpcio==1.80.0 \
        protobuf==7.34.1 \
        markdown==3.10.2 \
        werkzeug==3.1.8 \
        imageio==2.37.3 \
        tifffile==2025.5.10 \
        lazy-loader==0.5 \
	beautifulsoup4==4.14.3 \
        soupsieve==2.8.3 \
        PySocks==1.7.1 \
        certifi==2026.2.25 \
        charset-normalizer==3.4.7 \
        idna==3.11 \
        urllib3==2.6.3

RUN git clone https://github.com/fbcotter/pytorch_wavelets /opt/pytorch_wavelets && \
    cd /opt/pytorch_wavelets && \
    git checkout 9a0c507f04f43c5397e384bb6be8340169b2fd9a && \
    pip3 install . && \
    cd /

WORKDIR /workspace

CMD ["python3", "-c", "import torch; import cv2; import basicsr; import pytorch_wavelets; print('All modules OK'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"]
