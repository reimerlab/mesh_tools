FROM ubuntu:18.04 AS base
LABEL mantainer="Brendan Celii <brendanacelii@gmail.com>"
# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get install -y software-properties-common \
                       build-essential \
                       git \
                       git-lfs \
                       wget \
                       vim \
                       curl \
                       zip \
                       zlib1g-dev \
                       unzip \
                       pkg-config \
                       libblas-dev \
                       liblapack-dev \
                       python3-tk \
                       python3-wheel \
                       graphviz \
                       libhdf5-dev \
                       python3.8 \
                       python3.8-dev \
                       python3.8-distutils \
                       swig \
                       fish \
                       # necessary for building ipygraph (old docker guild)
                       # libxml2 \
                       # libxml2-dev \
                       # zlib1g-dev
                       &&\



    apt-get clean &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 &&\
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    python3 get-pip.py &&\
    rm get-pip.py &&\
    # best practice to keep the Docker image lean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Installing node
# ENV NODE_VERSION=18.0.0
# RUN apt install -y curl
# RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
# ENV NVM_DIR=/root/.nvm
# RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
# RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
# RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
# ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"
# RUN node --version
# RUN npm --version


# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     npm
# RUN npm install npm@latest -g && \
#     npm install n -g && \
#     n latest

# RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
#     && apt-get install -y nodejs

# RUN conda install -c conda-forge jupyterlab
# RUN conda install -c conda-forge nodejs

# Install essential Python packages
RUN pip3 install setuptools=='59.6.0' \
    && pip3 install --upgrade pip==21.3.1


RUN python3 -m pip --no-cache-dir install \
    anyio==3.7.1 \ 
    argon2-cffi==21.3.0 \ 
    argon2-cffi-bindings==21.2.0 \ 
    arrow==1.2.3 \ 
    asn1crypto==0.24.0 \ 
    asttokens==2.2.1 \ 
    async-lru==2.0.4 \ 
    attrs==23.1.0 \ 
    Babel==2.12.1 \ 
    backcall==0.2.0 \ 
    beautifulsoup4==4.12.2 \ 
    bleach==6.0.0 \ 
    bqplot==0.12.40 \ 
    certifi==2023.7.22 \ 
    cffi==1.15.1 \ 
    charset-normalizer==3.2.0 \ 
    comm==0.1.4 \ 
    contourpy==1.1.0 \ 
    cryptography==2.1.4 \ 
    cycler==0.11.0 \ 
    Cython==0.29.36 \ 
    debugpy==1.6.7.post1 \ 
    decorator==5.1.1 \ 
    defusedxml==0.7.1 \ 
    exceptiongroup==1.1.2 \ 
    executing==1.2.0 \ 
    fastjsonschema==2.18.0 \ 
    fonttools==4.42.0 \ 
    fqdn==1.5.1 \ 
    idna==3.4 \ 
    importlib-metadata==6.8.0 \ 
    importlib-resources==6.0.1 \ 
    ipydatawidgets==4.3.5 \ 
    ipykernel==6.25.1 \ 
    ipython==8.12.2 \ 
    ipython-genutils==0.2.0 \ 
    ipyvolume==0.6.3 \ 
    ipyvue==1.9.2 \ 
    ipyvuetify==1.8.10 \ 
    ipywebrtc==0.6.0 \ 
    ipywidgets==8.1.0 \ 
    isoduration==20.11.0 \ 
    jedi==0.17.2 \ 
    Jinja2==3.1.2 \ 
    json5==0.9.14 \ 
    jsonpointer==2.4 \ 
    jsonschema==4.19.0 \ 
    jsonschema-specifications==2023.7.1 \ 
    jupyter-contrib-core==0.4.2 \ 
    jupyter-contrib-nbextensions==0.7.0 \ 
    jupyter-events==0.7.0 \ 
    jupyter-highlight-selected-word==0.2.0 \ 
    jupyter-lsp==2.2.0 \ 
    jupyter-nbextensions-configurator==0.6.3 \ 
    jupyter_client==8.3.0 \ 
    jupyter_core==5.3.1 \ 
    jupyter_server==2.7.0 \ 
    jupyter_server_terminals==0.4.4 \ 
    jupyterlab==4.0.2 \ 
    jupyterlab-pygments==0.2.2 \ 
    jupyterlab-widgets==3.0.8 \ 
    jupyterlab_server==2.24.0 \ 
    keyring==10.6.0 \ 
    keyrings.alt==3.0 \ 
    kiwisolver==1.4.4 \ 
    lxml==4.9.3 \ 
    MarkupSafe==2.1.3 \ 
    matplotlib==3.7.2 \ 
    matplotlib-inline==0.1.6 \ 
    mistune==3.0.1 \ 
    nbclassic==1.0.0 \ 
    nbclient==0.8.0 \ 
    nbconvert==7.7.3 \ 
    nbformat==5.9.2 \ 
    nest-asyncio==1.5.7 \ 
    notebook==6.5.4 \ 
    notebook_shim==0.2.3 \ 
    numpy==1.24.4 \ 
    overrides==7.4.0 \ 
    packaging==23.1 \ 
    pandas==2.0.3 \ 
    pandocfilters==1.5.0 \ 
    parso==0.7.1 \ 
    pexpect==4.8.0 \ 
    pickleshare==0.7.5 \ 
    Pillow==10.0.0 \ 
    pkgutil_resolve_name==1.3.10 \ 
    platformdirs==3.10.0 \ 
    prometheus-client==0.17.1 \ 
    prompt-toolkit==3.0.39 \ 
    psutil==5.9.5 \ 
    ptyprocess==0.7.0 \ 
    pure-eval==0.2.2 \ 
    pycparser==2.21 \ 
    pycrypto==2.6.1 \ 
    Pygments==2.16.1 \ 
    PyGObject==3.26.1 \ 
    pyparsing==3.0.9 \ 
    python-apt==1.6.6 \ 
    python-dateutil==2.8.2 \ 
    python-json-logger==2.0.7 \ 
    pythreejs==2.4.2 \ 
    pytz==2023.3 \ 
    pyxdg==0.25 \ 
    PyYAML==6.0.1 \ 
    pyzmq==25.1.0 \ 
    referencing==0.30.2 \ 
    requests==2.31.0 \ 
    rfc3339-validator==0.1.4 \ 
    rfc3986-validator==0.1.1 \ 
    rpds-py==0.9.2 \ 
    Rtree==0.8.3 \ 
    SecretStorage==2.3.1 \ 
    Send2Trash==1.8.2 \ 
    simplejson==3.19.1 \ 
    six==1.11.0 \ 
    sniffio==1.3.0 \ 
    soupsieve==2.4.1 \ 
    stack-data==0.6.2 \ 
    terminado==0.17.1 \ 
    tinycss2==1.2.1 \ 
    tomli==2.0.1 \ 
    tornado==6.3.2 \ 
    traitlets==5.9.0 \ 
    traittypes==0.2.1 \ 
    typing_extensions==4.7.1 \ 
    tzdata==2023.3 \ 
    unattended-upgrades==0.1 \ 
    uri-template==1.3.0 \ 
    urllib3==2.0.4 \ 
    wcwidth==0.2.6 \ 
    webcolors==1.13 \ 
    webencodings==0.5.1 \ 
    websocket-client==1.6.1 \ 
    widgetsnbextension==4.0.8 \ 
    zipp==3.16.2


RUN jupyter nbextension enable --py --sys-prefix ipyvolume \
    && jupyter nbextension enable --py --sys-prefix widgetsnbextension

    #&& jupyter labextension install ipyvolume \
    #&& jupyter nbextension enable --py widgetsnbextension \
    #&& jupyter labextension install jupyter-webrtc \
    #&& jupyter labextension install @jupyter-widgets/jupyterlab-manager \ 

# -- installing datajoint and dotmotif (moved to the ) ====
# WORKDIR /
# RUN git clone https://github.com/aplbrain/grandiso-networkx.git \
#     && git clone https://github.com/celiibrendan/dotmotif.git
# ADD ./grandiso_patch_files/queues.py /grandiso-networkx/grandiso/queues.py
# RUN pip3 install /grandiso-networkx/ --no-cache-dir \ 
#     && pip3 install /dotmotif/ --no-cache-dir \
#     && pip3 install git+https://github.com/spapa013/datajoint-python.git --no-cache-dir


#-------------------------------------------------------------
FROM base AS mesh_lite

ADD ./python_bash_files/embree.bash /src/embree.bash
RUN pip3 install cython==0.29.36

RUN apt-get update && apt-get install -q -y \
    python3-rtree \
    cmake-curses-gui \
    libtbb-dev \
    libglfw3-dev \
    && chmod +x /src/embree.bash && /src/embree.bash

#-------------------------------------------------------------
FROM mesh_lite AS mesh_heavy

# --- cgal and meshlab dependencies----
RUN apt-get update && apt-get install -q -y \
    build-essential \
    g++ \
    libeigen3-dev \
    qt5-qmake \
    qtscript5-dev \
    libqt5xmlpatterns5-dev \
    libqt5opengl5-dev \
    assimp-utils \
    nano \
    xvfb \
    && rm -rf /var/lib/apt/lists/*


# CGAL Installation ########################################################

RUN add-apt-repository universe \
    && apt-get update \
    && apt-get -y install libboost-all-dev libgmp-dev libmpfr-dev libcgal-dev libboost-wave-dev libeigen3-dev

# MESHLAB Installation ########################################################

ARG QMAKE_FLAGS="-spec linux-g++ CONFIG+=release CONFIG+=qml_release CONFIG+=c++11 QMAKE_CXXFLAGS+=-fPIC QMAKE_CXXFLAGS+=-std=c++11 QMAKE_CXXFLAGS+=-fpermissive INCLUDEPATH+=/usr/include/eigen3 LIBS+=-L/meshlab/src/external/lib/linux-g++"
ARG MAKE_FLAGS="-j"

ENV PATH="/meshlab/src/distrib:${PATH}"
ENV PIP_ROOT_USER_ACTION=ignore
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

#### manually running the steps from https://hub.docker.com/r/hamzamerzic/meshlab/ ###

# add all of the lines that will compile the meshlabserver
WORKDIR /
RUN git clone https://github.com/hamzamerzic/vcglib -b devel \
    && git clone https://github.com/hamzamerzic/meshlab -b devel

#replace the certain files in vcglib and meshlab so will compile
RUN rm /vcglib/wrap/io_trimesh/import_nvm.h /vcglib/wrap/io_trimesh/import_out.h /meshlab/src/meshlab_mini.pro

ADD ./meshlab_patch_files/import_nvm.h /vcglib/wrap/io_trimesh/import_nvm.h
ADD ./meshlab_patch_files/import_out.h /vcglib/wrap/io_trimesh/import_out.h
ADD ./meshlab_patch_files/meshlab_mini.pro /meshlab/src/meshlab_mini.pro


# ----- this was not working so had to comment it out
#WORKDIR /meshlab/src/external
#RUN qmake -qt=5 external.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /meshlab/src/common
RUN qmake -qt=5 common.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /meshlab/src
RUN qmake -qt=5 meshlab_mini.pro $QMAKE_FLAGS && make $MAKE_FLAGS

WORKDIR /meshlab/src/meshlabplugins/filter_ao/
RUN qmake -qt=5 filter_ao.pro $QMAKE_FLAGS && make $MAKE_FLAGS


# --- installing cgal python wrappers
RUN . /etc/profile
ADD ./CGAL /src/CGAL
#ADD ./CGAL/cgal_skeleton_param /src/CGAL/cgal_skeleton_param 


# ---- for some reason this isnt installling correctly but doing from container works fine
RUN \
    pip3 install -e /src/CGAL/cgal_segmentation \
    && pip3 install -e /src/CGAL/cgal_skeleton \ 
    && pip3 install -e /src/CGAL/cgal_skeleton_param 


# FROM BASE AS TORCH_CPU

# RUN \
#     python3 -m pip --no-cache-dir install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
#     && python3 -m pip --no-cache-dir install torch-scatter -f https://data.pyg.org/whltorch-1.10.2+cpu.html \
#     && python3 -m pip --no-cache-dir install torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cpu.html \
#     && python3 -m pip --no-cache-dir install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
#     && python3 -m pip --no-cache-dir install \
#         torch-geometric \
#         tensorboardX \
        

# ---- enabling the jupyter lab launch ----
WORKDIR /

EXPOSE 8888

ADD ./jupyter/run_jupyter_unix.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter_unix.sh"]

