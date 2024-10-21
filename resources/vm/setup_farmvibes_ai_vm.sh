#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Update apt
sudo apt update

# Install git
sudo apt install git -y

# Install python
export DEBIAN_FRONTEND=noninteractive
sudo apt install python3 python-is-python3 python3-pip -y

# Install docker dependencies
sudo apt install ca-certificates gnupg lsb-release -y

# Add Docker’s official GPG key:
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# set up the repository. The first command creates the source list entry
(
cat << EOF
  deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable
EOF
) | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install our dependencies
sudo apt update
sudo apt install unzip -y


# Install docker engine
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io -y

# Obtain docker user
if [ "$EUID" -eq 0 ]; then
  export DOCKER_USER="${DOCKER_USER:-"azureuser"}"
else
  export DOCKER_USER="${DOCKER_USER:-$USER}"
fi

# Run docker without sudo
sudo usermod -aG docker $DOCKER_USER

# Run git-lfs install to restore large files
sudo apt install git-lfs -y
git lfs install
git lfs pull