FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    vim \
    wget

RUN groupadd -g 999 eunbinpark
RUN useradd -r -u 999 -g eunbinpark eunbinpark

RUN rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# set timezone
ENV TZ=Asia/Seoul
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt

SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["bash", "run.sh"]