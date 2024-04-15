# FROM gcr.io/kaggle-gpu-images/python:latest
# FROM gcr.io/kaggle-gpu-images/python:v128
FROM gcr.io/kaggle-gpu-images/python:v132

RUN touch /etc/apt/apt.conf.d/99verify-peer.conf \
    && echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

RUN apt-get update && \
    apt-get install -y \
    vim \
    git \
    byobu \
    htop \
    zip \
    unzip \
    sudo

RUN pip install omegaconf
RUN pip install "pysen[lint]"
# RUN pip install nbconvert==5.4.1 # for avoiding jupyter notebook 500 error.
RUN pip install iterative-stratification
RUN pip install -U nbqa
# jupyterlab, jupyterlab-vimは以下の組み合わせのバージョン以上だとvimのショートカットがおかしくなる
# 将来的に修正される可能性があるが、しばらくこのバージョンで様子見(2024/02/15)
RUN pip install jupyterlab==4.0.9
RUN pip install --upgrade jupyterlab-vim==4.1.0
RUN pip install git+https://github.com/rwightman/pytorch-image-models
RUN pip install -U isort # upgrade for nbqa
RUN pip install jupytext --upgrade
RUN pip install jupyterlab-code-formatter
RUN pip install polars
RUN pip install colorednoise
RUN pip install segmentation-models-pytorch
