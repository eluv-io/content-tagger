#!/bin/bash




models=(content-tagger shot asr celeb llava caption logo ocr speaker verticalvideo music breakdance qwen multilingual qwenjoe helmet hungarian player)

if [ "$1" = "" ]; then
    set -- "${models[@]}"
fi

declare -A status

for model in "$@"; do
    echo ------ "$model"
    status["$model"]="failed"
    podman pull "cr.elv/ml/${model}:latest" && podman tag "cr.elv/ml/${model}:latest" "localhost/${model}:latest" && status["$model"]="fetched"
    ## future: tag as elv-prod when we switch to that
done


outputstatus() {
    status="$1"
    shift
    commaspace=""
    for model in "$@"; do
        if [ "${status["$model"]}" == "$status" ]; then
            echo -n "${commaspace}${model}"
            commaspace=", "
        fi
    done
    echo
}

echo ====== status report

echo -n " failed: " ; outputstatus failed "$@"

echo -n "fetched: " ; outputstatus fetched "$@"

    
