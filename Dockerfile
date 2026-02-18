# Base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install base system and build tools
RUN apt update && apt install -y --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    git \
    cmake \
    build-essential \
    python3-pip \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install Project Dependencies
RUN apt update && apt install -y --no-install-recommends \
    libeigen3-dev \
    libboost-all-dev \
    libomp-dev \
    libpcl-dev \
    libyaml-cpp-dev \
    libflann-dev \
    liblz4-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    libgflags-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install Ceres Solver >= 2.1.0 from source (QuaternionManifold support)
WORKDIR /tmp
RUN git clone https://github.com/ceres-solver/ceres-solver.git && \
    cd ceres-solver && \
    git checkout 2.1.0 && \
    mkdir build && cd build && \
    cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF && \
    make -j$(nproc) && make install && \
    ldconfig && \
    cd / && rm -rf /tmp/ceres-solver

# Create non-root user
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set workspace structure
USER $USERNAME
WORKDIR /home/$USERNAME/workspace

RUN mkdir -p input output src

# Clone G-ED repository
WORKDIR /home/$USERNAME/workspace/src
RUN git clone https://github.com/robotics-upo/G-ED.git

# Go back to workspace root
WORKDIR /home/$USERNAME/workspace

CMD ["bash"]