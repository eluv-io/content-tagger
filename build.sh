#!/bin/bash

set -e

git submodule update --init --recursive

exec buildscripts/build_container.bash -t "content-tagger:${IMAGE_TAG:-latest}" . -f Dockerfile "$@"
