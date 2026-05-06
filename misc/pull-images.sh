#!/bin/bash
images=(content-tagger celeb qwen qwenjoe content-tagger shot llava ocr logo asr caption player highlight-composition summary highlights elv-vector-search)

if [ "$1" ]; then
    images=("$@")
fi

source=ml-004.eluvio:5100
tag=latest

for img in "${images[@]}"; do

    echo --------- "$img"
    echo podman pull "$source/$img:$tag" </dev/null
    podman tag "$source/$img:$tag" "localhost/$img:$tag"
done

tail=1
for img in "${images[@]}"; do        
    podman images "$img:$tag" | tail -n +$tail
    tail=2
done
