FROM ubuntu:18.04

ENV PYTHON_VERSION 3.7.0
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

# install pyenv for allowing to install any python
RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
 && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && $PYENV_ROOT/plugins/python-build/install.sh \
 && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
 && rm -rf $PYENV_ROOT

# install pip and locales
RUN apt-get -y update \
    && apt-get -y upgrade \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && pip install -U pip \
    && apt-get install -y sudo curl wget locales \
    && rm -rf /var/lib/apt/lists/* \
    && echo "ja_JP UTF-8" > /etc/locale.gen \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8
# install packages in requirements.txt
#WORKDIR /nlpbook
ADD requirements.txt ./
#/nlpbook
RUN pip install -r ./requirements.txt    # requirements.txtからパッケージのインストール

# below codes are referenced; https://system.blog.uuum.jp/entry/2019/10/18/110000
# mecabとmecab-ipadic-NEologdの導入; see here -> https://github.com/neologd/mecab-ipadic-neologd/blob/master/README.ja.md
RUN apt-get update \
    && apt-get install -y mecab \
    && apt-get install -y libmecab-dev \
    && apt-get install -y mecab-ipadic-utf8 \
    && apt-get install -y git \
    && apt-get install -y make \
    && apt-get install -y curl \
    && apt-get install -y xz-utils \
    && apt-get install -y file \
    && apt-get install -y sudo

# see here -> https://stackoverflow.com/questions/30716937/dockerfile-build-possible-to-ignore-error
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git ; exit 0
RUN cd mecab-ipadic-neologd \
    && bin/install-mecab-ipadic-neologd -n -y

RUN pip install mecab-python3


# nodejsの導入
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - \
    && sudo apt-get install -y nodejs

## JupyterLabの拡張機能
RUN jupyter labextension install @lckr/jupyterlab_variableinspector \
    && jupyter labextension install @jupyterlab/toc

# 自動整形
RUN pip install autopep8 \
    && pip install jupyterlab_code_formatter \
    && jupyter labextension install @ryantam626/jupyterlab_code_formatter \
    && jupyter serverextension enable --py jupyterlab_code_formatter
