FROM ubuntu:24.10
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y \ 
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \ 
        liblzma-dev \
        python3-pip \
        git-all \
        vim # && \ 
   # rm -fr /var/lib/apt/lists

# alias python2 to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Poetry
RUN mkdir -p /home/poetry && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/poetry python -


# Install PyEnv
RUN curl https://pyenv.run | bash


