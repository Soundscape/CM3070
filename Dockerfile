FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu22.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y apt-utils curl lsb-release libtinfo6 && \
    apt-get clean all

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app
COPY ./ ./

# Setup conda environment
RUN conda env create -q -f environment.yml && \
    echo "source activate final-project" > ~/.bashrc
ENV PATH /opt/conda/envs/final-project-api/bin:$PATH

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib/"
