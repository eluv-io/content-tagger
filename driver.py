#!/usr/bin/env python

import argparse
import requests
import subprocess
import json
import os
import traceback
from copy import deepcopy

from common_ml.utils.dictionary import nested_update

"""Convenience script for driving the tagger on bulk content."""

server = os.environ.get("TAGGERV2_HOST", "http://localhost:8086")

llava_prompt = "This is an image from a rugby match broadcast. Do not describe what people are wearing. Focus on the action and play depicted in the image. Describe the image in 2 sentences."
# will round robin between these models
llava_models = ["elv-llamavision:1", "elv-llamavision:2"]

assets_params = {"features": {"logo":{}, "ocr": {}}, "replace": False}

video_params = {
    "replace": False,
    "features": {
        "llava_brief": { 
            "model": {
                "fps": 0.1, "prompt": llava_prompt, 
                      "llama_endpoint": "http://localhost:11434",
                      "models": [ "elv-llamavision:1", "llama3.2-vision:latest"]  
            }
        }
    }
}

video_params = {
    "replace": False,
    "features": {
        "asr": {"stream": "audio" },
        "ocr": {},
        "shot": {},
        "llava": {"model": {"fps": 0.33, "prompt": llava_prompt} },
        "caption": {},
        "celeb": {},
        "logo": {}
    }
}
    

def get_auth(config: str, qhit: str) -> str:
    cmd = f"qfab_cli content token create {qhit} --update --config {config}"
    out = subprocess.run(cmd, shell=True, check=True, capture_output=True).stdout.decode("utf-8")
    token = json.loads(out)["bearer"]
    return token

def get_write_token(qhit: str, config: str) -> str:
    cmd = f"qfab_cli content edit {qhit} --config {config}"
    out = subprocess.run(cmd, shell=True, check=True, capture_output=True).stdout.decode("utf-8")
    write_token = json.loads(out)["q"]["write_token"]
    return write_token

def get_status(qhit: str, auth: str):
    res = requests.get(f"{server}/{qhit}/status", params={"authorization": auth})
    return response_force_dict(res)

def tag(contents: list, auth: str, assets: bool, start_time: float = None, end_time: float = None):
    if assets:
        params = deepcopy(assets_params)
    else:
        params = deepcopy(video_params)
    
    llama_models = ["elv-llamavision:1", "elv-llamavision:2"]
    for i, qhit in enumerate(contents):
        if assets:
            url = f"{server}/{qhit}/image_tag"
        else:
            url = f"{server}/{qhit}/tag"
        if "llava" in params["features"]:
            params["features"]["llava"]["model"]["model"] = llama_models[i % len(llama_models)]
        
        if start_time is not None:
            params["start_time"] = start_time

        if end_time is not None:
            params["end_time"] = end_time
                    
        res = requests.post(url, params={"authorization": auth}, json=params)
        print(response_force_dict(res))

def is_running(qhit: str, auth: str):
    res = requests.get(f"{server}/{qhit}/status", params={"authorization": auth})
    resdict = response_force_dict(res)
    if "error" in resdict:
        # No jobs started
        return False
    for stream, features in resdict.items():
        for feature, status in features.items():
            if status != "Completed":
                return True
    return False

def commit(write_token: str, config: str):
    cmd = f"qfab_cli content finalize {write_token} --config {config}"
    out = subprocess.run(cmd, shell=True, check=True, capture_output=True).stdout.decode("utf-8")
    print(out)

def response_force_dict(resp):
    try:
        return resp.json()
    except requests.exceptions.JSONDecodeError as e:
        return {
            "error": "could not parse json",
            "status": resp.status_code,
            "content": resp.content
        }

def finalize(qhit: str, config: str, do_commit: bool):
    auth_token = get_auth(config, qhit)
    write_token = get_write_token(qhit, config)
    finalize_url = f"{server}/{qhit}/finalize?authorization={auth_token}"
    resp = requests.post(finalize_url, params={"write_token": write_token, "replace": "true"})
    respdict = response_force_dict(resp)
    print(respdict)
    if do_commit and "error" not in respdict:
        commit(write_token, config)

    return write_token

def aggregate(qhit: str, config: str, do_commit: bool):
    auth_token = get_auth(config, qhit)
    write_token = get_write_token(qhit, config)
    finalize_url = f"{server}/{qhit}/aggregate?authorization={auth_token}"
    resp = requests.post(finalize_url, params={"write_token": write_token, "replace": "true"})
    respdict = response_force_dict(resp)
    print(respdict)
    if do_commit and "error" not in respdict:
        commit(write_token, config)

    return write_token

def main():
    if args.contents:
        print("reading contents...")
        with open(args.contents, 'r') as f:
            contents = [line.strip() for line in f.readlines()]
            if len(contents) == 0:
                raise ValueError("No contents found in file.")
    else:
        contents = [args.iq]

    print("getting auth...")
    auth = get_auth(args.config, contents[0])
    
    print("Command (t)ag, (s)tatus, (qs)quickstatus, (f)inalize, (a)ggregate? ")
    while True:
        try:
            user_input = input("> ")  # Wait for user input
            if user_input == "tag" or user_input == "t":
                tag(contents, auth, args.assets) ## end_time=20.5)
            elif user_input == "qs":
                for qhit in contents:
                    status = get_status(qhit, auth)
                    for imgorvid, models in status.items():
                        for model, stat in models.items():
                            print("[%9s] %-32s / %s: %s" % (stat.get("tagging_progress", ""), qhit, f"({imgorvid}) {model}", stat.get("status", "??") ) )
            elif user_input in [ "status", "s" ]:
                statuses = {}
                for qhit in contents:
                    status = get_status(qhit, auth)
                    statuses[qhit] = status
                    print(qhit, json.dumps(status, indent=2))
                os.makedirs("driver_workdir", exist_ok=True)
                with open("driver_workdir/status.json", "w") as statfile:
                    statfile.write(json.dumps(statuses, indent = 2))
            elif user_input in [ "finalize", "f"]:
                for qhit in contents:
                    finalize(qhit, args.config, args.commit)
            elif user_input in [ "agg", "aggregate"]:
                for qhit in contents:
                    aggregate(qhit, args.config, args.commit)
            elif user_input in [ "quit", "exit"]:
                break
            else:
                print(f"Invalid command: {user_input}")
        except KeyboardInterrupt:
            print("")
            break
        except EOFError:
            print("")
            break
        except:
            print(traceback.format_exc())

    print("Exiting")
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tag")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--contents")
    group.add_argument("-q", "--iq")    
    parser.add_argument("--assets", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--tag_config", default="{}")
    parser.add_argument("--audio_stream", default="audio")
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    main()
