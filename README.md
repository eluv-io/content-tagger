# Content-Tagger
A server for running individual containerized tagger models and publishing tags.

## Prerequisites

1. Podman with nvidia-toolkit enabled
2. Python
3. A podman image for each model you want to run. See the individual model repos for more information. 
4. A gpu, cpu inference is not yet supported. 

## Setup w/ conda

1. Start podman API socket `systemctl --user start podman.socket`
2. Create new conda environment `conda create -n tagger-services python=3.10`
3. Activate the environment `conda activate tagger-services`
4. Install dependencies `pip install .`

## Usage

### Start server

1. `python server.py --port <PORT_NUMBER>`

### Call API

Check out the swagger docs in api.yaml