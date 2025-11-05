#!/usr/bin/env python
import readline
import time
import signal
import sys

interrupt = False    ## can be removed and calculated from other values...
lastInput = ""
lastInputTime = 0
timeout_value = 1000

def interrupted(signum, stack):
    global interrupt, lastInput, lastInputTime, timeout_value
    buf = readline.get_line_buffer()
    local_timeout_value = timeout_value
    if local_timeout_value is None:
        return
    elif (buf != lastInput):
        ## print("buf changed", buf)
        lastInput = buf
        lastInputTime = time.time()
    elif time.time() - lastInputTime > local_timeout_value:
        ## print("buf did not change and time was up")
        interrupt = True
        raise Exception("Input timed out.")
    
    ## reset alarm if buf changed OR time was less than timeout
    signal.alarm(timeout_value)

signal.signal(signal.SIGALRM, interrupted)

def get_input(prompt, timeout = None):
    global interrupt, lastInputTime, timeout_value
    
    interrupt = False
    timeout_value = timeout
    lastInputTime = time.time()

    if timeout: signal.alarm(2)

    try:
        line = input(prompt)
        timeout_value = None
        lastInputTime = time.time()
        return line
    except Exception as e:
        if interrupt:
            return None
        raise e
        
import argparse
import requests
import subprocess
import json
import os
import re
import traceback
from copy import deepcopy

from common_ml.utils.dictionary import nested_update

"""Convenience script for driving the tagger on bulk content."""

server = os.environ.get("TAGGERV2_URL", "http://localhost:8086")
written = {}

llava_prompt = "This is an image from a rugby match broadcast. Do not describe what people are wearing. Focus on the action and play depicted in the image. Describe the image in 2 sentences."
# will round robin between these models
llava_models = ["elv-llamavision:1", "elv-llamavision:2"]

assets_params = {"features": {"logo":{}, "ocr": {}}, "replace": False}

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
    if qhit.startswith("tqw"):
        return qhit
    
    cmd = f"qfab_cli content edit {qhit} --config {config}"
    out = subprocess.run(cmd, shell=True, check=True, capture_output=True).stdout.decode("utf-8")
    write_token = json.loads(out)["q"]["write_token"]
    return write_token

def allstatus(auth: str):
    res = requests.get(f"{server}/allstatus", params={"authorization": auth})
    return response_force_dict(res)

def get_status(qhit: str, auth: str):
    res = requests.get(f"{server}/{qhit}/status", params={"authorization": auth})
    return response_force_dict(res)

def tag(contents: list, auth: str, assets: bool, params: dict, start_time: float = None, end_time: float = None):
    
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

        print(json.dumps(params, indent=2))
        res = requests.post(url, params={"authorization": auth}, json=params)
        print(response_force_dict(res))

        time.sleep(float(os.environ.get("TAGGERV2_START_SLEEP", 0)))

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

def write(qhit: str, config: str, do_commit: bool, force = False, leave_open = False):
    auth_token = get_auth(config, qhit)
    write_token = get_write_token(qhit, config)
    write_url = f"{server}/{qhit}/finalize?authorization={auth_token}&force={force}"
    resp = requests.post(write_url, params={"write_token": write_token, "replace": "true", "leave_open": leave_open})
    respdict = response_force_dict(resp)
    print(respdict)
    if do_commit and "error" not in respdict:
        commit(write_token, config)

    return write_token

def aggregate(qhit: str, config: str, do_commit: bool):
    auth_token = get_auth(config, qhit)
    write_token = get_write_token(qhit, config)
    aggregate_url = f"{server}/{qhit}/aggregate?authorization={auth_token}"
    resp = requests.post(aggregate_url, params={"write_token": write_token, "replace": "true"})
    respdict = response_force_dict(resp)
    print(respdict)
    if do_commit and "error" not in respdict:
        commit(write_token, config)

    return write_token

def write_all(contents: list, config: str, do_commit: bool, force = False):
    for qhit in contents:
        if qhit in written:
            print(f"{qhit} already written, clearwritten to clear list")
            continue

        print(f"Finalizing {qhit} force = {force}")
        try:
            leave_open = False
            if qhit.startswith("tqw"):
                leave_open = True
                do_commit = False
            
            write(qhit, config, do_commit, force, leave_open)
            written[qhit] = True ### xxx store a hash of written job IDs
            
        except Exception as e:
            print(f"{e} while finalizing {qhit}")

def stop(qhit: str, auth: str, features: list[str]):
    """
    Stops the tagging process for a specific track of a given content (iq).

    Args:
        iq (str): The content identifier.
        auth (str): Authorization token.
        features (list): The tracks to stop tagging for.
    """

    params = {"authorization": auth}
    for feature in features:
        url = f"{server}/{qhit}/stop/{feature}"
        try:
            res = requests.post(url, params=params)
            if res.status_code == 200:
                print(f"Successfully stopped tagging for {qhit} on track {feature}.")
            else:
                print(f"Failed to stop tagging for {qhit} on track {feature}: {res.status_code} {res.text}")
        except Exception as e:
            print(f"Error while stopping tagging for {qhit} on track {feature}: {e}")

def list_models():
    print("getting model list:")
    modresp = requests.get(f"{server}/list")
    modresp.raise_for_status()
    return modresp.json()

def help():
    print("""
t,tag [iq_regex] [model]        tag content
                                if iq_regex specified, only tag matching
                                if model specified, only start that model
stop [iq] [model]               stop tagging
                                if iq given, only stop for that iq (must be full iq, not regex)
                                if model given, only stop for that model
s,status                        show status
qs [regex]                      quick status, if regex given, match output only containing regex
list                            list models tagger knows about
cw,clearwritten                 clear the "written" state to allow re-writing
h,help                          this help""")
    
def main():
    global written
    
    if args.tag_config != "":
            tag_config = args.tag_config
    else:
        if args.assets:
            tag_config = assets_params
        else:
            tag_config = video_params
    
    if args.tag_config.startswith('@'):
        conffile = args.tag_config[1:]
        print("reading tag config...")
        with open(conffile, "r") as conf:
           tag_config = json.load(conf)

    if args.contents:
        print("reading contents...")
        with open(args.contents, 'r') as f:
            contents = [line.strip() for line in f.readlines()]
            if len(contents) == 0:
                raise ValueError("No contents found in file.")
    else:
        contents = [] #args.iq]

    contents = contents + args.iq
        
    models = list_models()
    print("models:" , models)
    
    print("getting auth...")
    auth = get_auth(args.config, contents[0])

    end_time = None
    if args.end_time is not None: end_time = int(args.end_time)
    start_time = int(args.start_time)
    
    quickstatus_watch = None

    tty = sys.stdin.isatty()
    if tty:
        help()
        
    while True:
        try:
            if tty:
                timeout = None
                if quickstatus_watch is not None:
                    timeout = 60
                user_line = get_input(f"{server} > ", timeout = timeout)  # Wait for user input
            else:
                user_line = input("")
                
            if (quickstatus_watch and user_line is None):
                user_line = quickstatus_watch
                print("[auto quickstatus]")
            elif user_line != "" and not tty:
                ## echo the command if they are being piped in from a script...
                print("command: " + user_line)
                
            user_split = re.split(r" +", user_line)
            user_input = user_split[0]
            
            reset_quickstatus = True
            
            if user_input in [ "statusall", "sall"]:
                res = allstatus(os.environ.get("ADMIN_TOKEN", ""))
                print(json.dumps(res, indent=2))
            elif user_input in [ "status", "s"]:
                reset_quickstatus = False
                statuses = {}
                for qhit in contents:
                    if len(user_input) > 1:
                        if not re.search(user_input[1], qhit): continue                        
                    status = get_status(qhit, auth)
                    statuses[qhit] = status
                    print(qhit, json.dumps(status, indent=2))
                os.makedirs("rundriver", exist_ok=True)
                with open("rundriver/status.json", "w") as statfile:
                    statfile.write(json.dumps(statuses, indent = 2))
            elif user_input in [ "finalize", "f" ]:
                print("it's called 'write' now (to avoid confusion over what it does)")
            elif user_input in [ "list", "l" ]:
                models = list_models()
                print("models:", models)
            elif user_input in [ "cw", "clearwritten"]:
                iqsub = None
                if len(user_split) > 1:
                    iqsub = user_split[1]

                new_written = {}
                if iqsub:
                    for iq, state in written.items():
                        if re.search(iqsub, iq):
                            print(iq, "cleared")
                        else:
                            print(iq, "written")
                            new_written[iq] = written[iq]
                        
                written = new_written
            elif user_input in [ "stop" ]:
                if len(user_split) < 2:
                    print("must specify iq and optionally tag track")
                    continue
                iq = user_split[1]
                if len(user_split) > 2:
                    tracks = [user_split[2]]
                else:
                    tracks = models
                stop(iq, auth, tracks)
            elif user_input in [ "tag" , "t", "replacetag" ]:
                this_tag_config = deepcopy(tag_config)
                if user_input == "replacetag":
                    this_tag_config['replace'] = True
                iqsub = None
                if len(user_split) > 1:
                    iqsub = user_split[1]
                if len(user_split) > 2:
                    track = user_split[2]
                    this_tag_config['features'] = { track: tag_config['features'][track] }
                contentsub = [ x for x in contents if iqsub == None or re.search(iqsub, x) ]    
                tag(contentsub, auth, args.assets, this_tag_config,
                    start_time = start_time, end_time = end_time)
            elif user_input.startswith("+") or user_input.startswith("-"):

                val = user_input[1:]
                val = float(val) * 60
                if user_input.startswith("-"): val = val * -1
                    
                if end_time is None:
                    end_time = start_time + val
                else:                                        
                    end_time = end_time + int(val)
                    start_time = start_time + int(val)
                                    
                h = int(end_time / 3600)
                m = int(end_time / 60) % 3600
                s = int(end_time) % 60
                hms = "%d:%02d:%02d" % (h, m, s)
                h = int(start_time / 3600)
                m = int(start_time / 60) % 3600
                s = int(start_time) % 60
                hmss = "%d:%02d:%02d" % (h, m, s)
                print(f'[{start_time}-{end_time}] [{hmss} - {hms}]')                
            elif user_input == "qs":
                reset_quickstatus = False
                if len(user_split) > 1 and user_split[1] == "watch":
                    quickstatus_watch = " ".join(["qs"] + user_split[2:])
                    print("quickstatus on, command: " + quickstatus_watch)
                elif len(user_split) > 1 and user_split[1] == "off":
                    print("quickstatus off")
                    quickstatus_watch = None
                else:
                    for qhit in contents:
                        quick_status(auth, qhit, " ".join(user_split[1:]))
            elif user_input in [ 'reverse' ]:
                contents.reverse()
                print("First element:", contents[0])
            elif user_input in [ "write", "w" ]:
                contentsub = contents
                if len(user_split) > 1:
                    contentsub = user_split[1:]
                write_all(contentsub, args.config, args.commit, force = False)
            elif user_input in [ "forcewrite" ]:
                contentsub = contents
                if len(user_split) > 1:
                    contentsub = user_split[1:]
                write_all(contentsub, args.config, args.commit, force = True)
            elif user_input in [ "agg", "aggregate"]:
                contentsub = contents
                if len(user_split) > 1:
                    contentsub = user_split[1:]
                for qhit in contentsub:
                    aggregate(qhit, args.config, args.commit)
            elif user_input in [ "quit", "exit"]:
                break
            elif user_input in [ "h", "help"]:
                help()
            elif user_input == "":
                reset_quickstatus = False
            else:
                reset_quickstatus = False
                print(f"Invalid command: {user_input}")

            if reset_quickstatus and quickstatus_watch:
                quickstatus_watch = None
                print("[auto quickstatus turned off]")
                
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

def quick_status(auth, qhit, filter = None):
    if filter == "": 
        filter = None
    status = get_status(qhit, auth)
    if status.get("error", None):
        line = "[%9s] %-32s / %s: %s" % ("", qhit, "err", status['error']) 
        if filter is None or re.search(filter, line):
            print(line, flush = True)
        return
    for imgorvid, models in status.items():
        for model, stat in models.items():
            line =  "[%9s] %-32s / %s: %s" % (stat.get("tagging_progress", ""), qhit, f"({imgorvid}) {model}", stat.get("status", "??") ) 
            if filter is None or re.search(filter, line):
                print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic tag driver")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--contents", help="filename with list of contents (iq's) to tag")
    group.add_argument("-q", "--iq", nargs='*', default=[], help="content (iq) to tag (specified directly)")    
    parser.add_argument("--assets", action="store_true", help="if set, tag assets instead of videos")
    parser.add_argument("--config", help="fabric config file to use for making tokens")
    parser.add_argument("--tag-config", default="", help="Tagger config json.  Use @ to read a file")
    ##parser.add_argument("--audio-stream", default="audio", help="which audio stream to tag")
    parser.add_argument("--commit", "--finalize", action="store_true", help="if set, commit (finalize) on fabric after writing on tagger")
    parser.add_argument("--start-time", help="start time in seconds", default = 0)
    parser.add_argument("--end-time", help="end time in seconds", default = None)
    args = parser.parse_args()
    main()
