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

## Podman (Docker Image)

### Dependencies

1. Podman with nvidia toolkit enabled installed on machine
2. Access to qluvio repo via ssh key

#### Add ssh keys to ssh-agent
`ssh-add` (on personal machine)

**NOTE**: if you are on a remote server, either you should have your ssh key on the remote server and run `ssh-add` there, or you should run it on your personal machine and verify that you are connected with agent forwarding enabled.

#### Build image
`./build.sh`

### Run as container

You will also need a `tagger-config.yml` file in the data directory,
which in the example is `data` under the current directory.

Note that you need to specify the same path inside and outside of the
container for the data directory, as docker images are started with
mountpoints specified.  (Since the podman socket is passed in, the
tagger starts container jobs on the host's podman so the container
needs to understand the host directory layout; mounting under the same
path in the container insures this.)  For a similar reason you need to
pass all GPUs; GPUs are passed to containers using host GPU numbers so
the tagger container needs all GPUs so the GPU numbers are right.

`podman run --device nvidia.com/gpu=all -p 8086:8086 -v /run/user/$(id -u)/podman:/run/user/0/podman -v $(pwd)/data:$(pwd)/data content-tagger --directory $(pwd)/data --host 0.0.0.0`

## Usage

### Start server

1. `python server.py --port <PORT_NUMBER>`

### Call API

Check out the swagger docs in api.yaml
