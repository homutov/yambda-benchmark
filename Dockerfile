# cuda11.8 due to implicit
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel@sha256:e567ea2642d4e25ef647e9872d09980613a7b1ecc8d2973e339c60a07343046f

WORKDIR /yambda

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsuitesparse-dev \
        build-essential \
        git


ENV CUDACXX=/usr/local/cuda/bin/nvcc \
    SUITESPARSE_INCLUDE_DIR=/usr/include/suitesparse \
    SUITESPARSE_LIBRARY_DIR=/usr/lib

RUN pip install implicit

RUN git clone https://github.com/glami/sansa.git  && cd sansa && pip install . && cd ..

COPY . .
RUN pip install -e .
