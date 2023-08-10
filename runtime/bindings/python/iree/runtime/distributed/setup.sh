#!/bin/bash

set -e

distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget -O /tmp/cuda-keyring_1.0-1_all.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i /tmp/cuda-keyring_1.0-1_all.deb
sudo apt update
# For CMake to find CUDA when using LLD.
sudo apt -y install lld

sudo apt -y install libopenmpi-dev
sudo apt -y install libnccl-dev=2.18.1-1+cuda12.1
pip install mpi4py jax[cpu]
