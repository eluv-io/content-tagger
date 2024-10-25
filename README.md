# Tagger Services
A server for running individual containerized tagger models and publishing tags.

## Prerequisites

1. Podman with nvidia-toolkit enabled
2. Python
3. Built podman containers for all models you want to run. See individual model repos for more information.

## Setup w/ conda

1. Start podman API socket `systemctl --user start podman.socket`
2. `conda create -n tagger-services python=3.10`
3. `conda activate tagger-services`
4. `pip install .`

## Usage

### Start server

1. `python server.py --port <PORT_NUMBER>`

### Call API

1. Check available services `curl "http://localhost:<PORT_NUMBER>/list"`
2. Call tagger `curl -X POST "http://localhost:<PORT_NUMBER>/tag?authorization=$AUTH" -d '{"qid":"iq__42WgpoYgLTyyn4MSTejY3Y4uj81o", "features":["asr", "caption"]}' -H "Content-Type: application/json" | jq`
3. With specific start/end time (in seconds) `curl -X POST "http://localhost:<PORT_NUMBER>/tag?authorization=$AUTH" -d '{"qid":"iq__42WgpoYgLTyyn4MSTejY3Y4uj81o", "features":["asr", "caption"], start_time:100, end_time: 200}' -H "Content-Type: application/json" | jq`
4. On a specific stream, if default streams "audio" or "video" are not available `curl -X POST "http://localhost:<PORT_NUMBER>/tag?authorization=$AUTH" -d '{"qid":"iq__42WgpoYgLTyyn4MSTejY3Y4uj81o", "features":["asr"], "stream":"english_stereo"}' -H "Content-Type: application/json" | jq`