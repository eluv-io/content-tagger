#!/bin/bash

## the models
models=(model-asr model-player model-caption model-celeb model-llava model-logo model-ocr model-shot)

## default test GPU to 0 if not specified
ELV_MODEL_TEST_GPU_TO_USE="${ELV_MODEL_TEST_GPU_TO_USE:-0}"
export ELV_MODEL_TEST_GPU_TO_USE


if [ "$1" == "--update" ] || [ "$1" = "-u" ]; then
    update=true
fi

GIT_PAGER=cat
export GIT_PAGER

git_recent() {
    git for-each-ref --sort=-committerdate --color=auto --count=25 refs/heads refs/remotes \
        --format='%(refname:short)|%(HEAD)%(color:yellow)%(refname:short)|%(color:green)%(committerdate:relative)|%(color:magenta)%(authorname)%(color:reset)|%(color:blue)%(subject)' --color=always --count=${count:-20} | \
        sed -re 's/ago[|]/|/' | \
        column -ts '|'
}

cd ..

for repo in "${models[@]}"; do
    if [ ! -d "$repo" ]; then
        echo CLONE: $repo  -------------
        git clone git@github.com:eluv-io/$repo
    elif [ "$update" = "true" ]; then
        echo UPDATE: $repo  -------------
        (cd "$repo" && git fetch && git_recent && git pull && git status --porcelain )
    fi    
done

declare -A status

for repo in "${models[@]}"; do
    yes ========================= | head -20    
    echo "========================= $repo BUILD"
    
    (cd "$repo" ; ./build.sh ) || status[$repo]="build failed"
done

for repo in "${models[@]}"; do
    yes ========================= | head -5
    
    if [ "${status[$repo]}" = "build failed" ]; then
        echo "========================= $repo FAILED BUILD, no test"
    else
        echo "========================= test.sh"
        echo $repo BUILD -------------
    
        if (cd "$repo" ; ./test.sh ); then
            status[$repo]="success"
        else
            status[$repo]="test failed"
        fi
    fi
done

for repo in "${!status[@]}"
do
    printf "%14s: %s\n" "$repo" "${status[$repo]}"
done
