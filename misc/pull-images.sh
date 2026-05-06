#!/bin/bash
images=(content-tagger asr celeb qwen qwenjoe speaker shot llava ocr logo caption player highlight-composition summary highlights elv-vector-search)

if [ "$1" ]; then
    images=("$@")
fi

exec < /dev/null

source=cr.elv/ml
tag=latest

for img in "${images[@]}"; do

    echo --------- "$img"
    podman pull "$source/$img:$tag" && podman tag "$source/$img:$tag" "localhost/$img:$tag"
done

tail=1
for img in "${images[@]}"; do        
    podman images "$img:$tag" | tail -n +$tail
    tail=2
done
