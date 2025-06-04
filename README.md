# Content-Tagger
A server for running individual containerized tagger models and publishing tags.

## Features

1. Server for orchestrating tagging jobs across multiple containerized models. 
2. Supports concurrent tagging across multiple tenants. 
3. Splits jobs across all available GPUs and implements simple queing when there are more jobs than GPUs. 
4. Offers an endpoint for finalizing tags by publishing to the Fabric. 

## Prerequisites

1. Podman with nvidia-toolkit enabled
2. Python
3. A podman image for each model you want to run. See the individual model repos for more information. 
4. A gpu, cpu inference is not yet supported. 
5. ssh key with github access to eluv-io

## Setup w/ conda

1. Start podman API socket `systemctl --user start podman.socket`
2. Add private keys to ssh agent (if you are on remote server): `ssh-add` 
3. Create new conda environment `conda create -n tagger-services python=3.10`
4. Activate the environment `conda activate tagger-services`
5. Install dependencies `pip install .`

## Usage

### Start server

1. `python server.py --port <PORT_NUMBER>`

### Call API

Check out the swagger docs in api.yaml