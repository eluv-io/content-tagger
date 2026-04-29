# Content-Tagger
HTTP server for running tagging containers against content-fabric objects

## Features

1. Resource aware orchestration of concurrent tagging jobs across multiple containerized models. 
2. Fetches media from the content fabric for tagging. Supports static vod video/audio parts, image assets, live video/audio segments
3. Uploads tags to the tagstore: see https://ai.contentfabric.io/tagstore/docs

## API

Offers endpoints for managing tagging LROs. See https://ai.contentfabric.io/tagging-live/docs

## Install Prerequisites

1. Podman with nvidia-toolkit enabled
2. One or more podman tagger images installed locally
3. Podman API socket running `systemctl --user start podman.socket`

### Run Tagger with podman

#### Build image
`./build.sh`

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

#### Run server

`podman run --device nvidia.com/gpu=all -p 8086:8086 -v /run/user/$(id -u)/podman:/run/user/0/podman -v $(pwd)/data:$(pwd)/data content-tagger --directory $(pwd)/data --host 0.0.0.0`

### Run Tagger with python & conda

1. Create and activate new conda environment `conda create -n tagger python=3.10 && conda activate tagger`
2. Install dependencies `pip install .`
3. `python server.py --port PORT_NUMBER`

### Development & Testing

Check out the `architecture.md` files for general design

#### Run unit tests

`pytest tests`

#### Run integration tests

Many of the tests are currently configured read-only against live content objects in several tenancies. Enable these extra tests as follows:

1. `cp tests/integration_private_keys_example.json tests/integration_private_keys.json`
2. Fill in the private keys for each of the required tenancies.
3. `./tests/refresh_tokens.sh`
4. `pytest tests` should now run the full test suite