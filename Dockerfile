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
        vim 

# alias python2 to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install PyEnv
RUN curl https://pyenv.run | bash

RUN /bin/bash -c "echo 'export PATH=\"/\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"

RUN /bin/bash -c "echo 'export PYENV_ROOT=\"\$HOME/.pyenv\"' >> ~/.bashrc; \
                  echo '[[ -d \$PYENV_ROOT/bin ]] && export PATH=\"\$PYENV_ROOT/bin:\$PATH\"' >> ~/.bashrc; \
                  echo 'eval \"\$(pyenv init -)\"' >> ~/.bashrc"

ARG PYTHON_VER="3.10.14"

RUN /bin/bash -c "\$HOME/.pyenv/bin/pyenv install ${PYTHON_VER}"

