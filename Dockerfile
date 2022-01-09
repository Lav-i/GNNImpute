FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 git\
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

SHELL [ "/bin/bash", "--login", "-c" ]

WORKDIR /GNNImpute

COPY . /GNNImpute

RUN conda init bash \
    && conda create -n gnnimpute python=3.6 \
    && /usr/local/envs/gnnimpute/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && /usr/local/envs/gnnimpute/bin/pip install -r requirements.txt

RUN echo "source activate gnnimpute" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
