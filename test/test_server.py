from quick_test_py import Tester
from argparse import ArgumentParser
from elv_client_py import ElvClient
import os
from typing import List, Callable
from dataclasses import asdict
from loguru import logger
import requests
import time
import atexit
import subprocess
import json
import shutil

from common_ml.utils.metrics import timeit

from config import config

test_objects = {"vod": "iq__3C58dDYxsn5KKSWGYrfYr44ykJRm", "assets": "iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2", "legacy_vod": "hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU"}

def get_auth(qid: str) -> str:
    return os.getenv(f"AUTH_{qid}")

def get_write_token(qid: str) -> str:
    return os.getenv(f"WRITE_{qid}")

def postprocess_response(res: dict):
    for stream in res:
        for model in res[stream]:
            del res[stream][model]['tag_job_id']
            del res[stream][model]['time_running']
            del res[stream][model]['tagging_progress']
    return res

def test_server(port: int) -> List[Callable]:
    filedir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Starting server on port {port}")
    server_out = open(os.path.join(filedir, 'server_out.log'), 'w')
    process = subprocess.Popen(['python', os.path.join(filedir, "../server.py"), '--port', str(port)], stdout=server_out, stderr=server_out)
    time.sleep(2)
    def cleanup():
        logger.info("Cleaning up")
        server_out.close()
        process.terminate()

    atexit.register(cleanup)

    assets_auth = get_auth(test_objects['assets'])
    video_auth = get_auth(test_objects['vod'])
    legacy_vod_auth = get_auth(test_objects['legacy_vod'])
    
    parts_path = os.path.join(config["storage"]["parts"], test_objects['vod'])
    image_path = os.path.join(config["storage"]["images"], test_objects['assets'])
    legacy_vod_path = os.path.join(config["storage"]["parts"], test_objects['legacy_vod'])
    
    shutil.rmtree(parts_path, ignore_errors=True)
    shutil.rmtree(image_path, ignore_errors=True)
    shutil.rmtree(legacy_vod_path, ignore_errors=True)
    
    vod_tags_path = os.path.join(config["storage"]["tags"], test_objects['vod'], "video")
    image_tags_path = os.path.join(config["storage"]["tags"], test_objects['assets'], "image")
    legacy_vod_tags_path = os.path.join(config["storage"]["tags"], test_objects['legacy_vod'], "video")
    
    shutil.rmtree(vod_tags_path, ignore_errors=True)
    shutil.rmtree(image_tags_path, ignore_errors=True)
    shutil.rmtree(legacy_vod_tags_path, ignore_errors=True)

    # test cases
    def test_tag():
        res = []
        video_status_url = f"http://localhost:{port}/{test_objects['vod']}/status?authorization={video_auth}"
        response = requests.get(video_status_url)
        assert response.status_code == 404, response.text
        res.append(response.json())

        tag_url = f"http://localhost:{port}/{test_objects['vod']}/tag?authorization={video_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_gpu": {"model":{"tags":["hello1", "hello2"]}}, "shot":{}}, "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(1)
        response = requests.get(video_status_url)
        res.append(postprocess_response(response.json()))

        tag_url = f"http://localhost:{port}/{test_objects['vod']}/tag?authorization={video_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_cpu": {"model":{"tags":["a", "b", "a"], "allow_single_frame":False}}}, "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(10)

        response = requests.get(video_status_url)
        res.append(postprocess_response(response.json()))

        image_status_url = f"http://localhost:{port}/{test_objects['assets']}/status?authorization={assets_auth}"

        tag_url = f"http://localhost:{port}/{test_objects['assets']}/image_tag?authorization={assets_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_gpu": {"model":{"tags":["hello1"]}}}, "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(1)
        response = requests.get(image_status_url)
        res.append(postprocess_response(response.json()))

        tag_url = f"http://localhost:{port}/{test_objects['assets']}/image_tag?authorization={assets_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_cpu": {"model":{"tags":["hello2"]}}}, "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(10)
        response = requests.get(image_status_url)
        res.append(postprocess_response(response.json()))
        
        tag_url = f"http://localhost:{port}/{test_objects['legacy_vod']}/tag?authorization={legacy_vod_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_gpu": {"model":{"tags":["hello1"]}}, "shot":{}}, "start_time":60, "end_time":180, "replace": False})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(5)
        status_url = f"http://localhost:{port}/{test_objects['legacy_vod']}/status?authorization={legacy_vod_auth}"
        response = requests.get(status_url) 
        res.append(postprocess_response(response.json()))
        
        logger.debug("Waiting for server to finish tagging")
        time.sleep(120)

        response = requests.get(video_status_url).json()
        for stream in response:
            for model in response[stream]:
                assert response[stream][model]['status'] == 'Completed', response
        res.append(postprocess_response(response))

        response = requests.get(image_status_url).json()
        for stream in response:
            for model in response[stream]:
                assert response[stream][model]['status'] == 'Completed', response
        res.append(postprocess_response(response))

        video_tags_path = os.path.join(config["storage"]["tags"], test_objects['vod'], "video")
        for tag in sorted(os.listdir(os.path.join(video_tags_path, "dummy_gpu"))):
            with open(os.path.join(video_tags_path, "dummy_gpu", tag)) as f:
                res.append(json.load(f))
        for tag in sorted(os.listdir(os.path.join(video_tags_path, "dummy_cpu"))):
            with open(os.path.join(video_tags_path, "dummy_cpu", tag)) as f:
                res.append(json.load(f))

        image_tags_path = os.path.join(config["storage"]["tags"], test_objects['assets'], "image")
        for tag in sorted(os.listdir(os.path.join(image_tags_path, "dummy_gpu"))):
            with open(os.path.join(image_tags_path, "dummy_gpu", tag)) as f:
                res.append(json.load(f))
        for tag in sorted(os.listdir(os.path.join(image_tags_path, "dummy_cpu"))):
            with open(os.path.join(image_tags_path, "dummy_cpu", tag)) as f:
                res.append(json.load(f))
                
        legacy_vod_tags_path = os.path.join(config["storage"]["tags"], test_objects['legacy_vod'], "video")
        for tag in sorted(os.listdir(os.path.join(legacy_vod_tags_path, "dummy_gpu"))):
            with open(os.path.join(legacy_vod_tags_path, "dummy_gpu", tag)) as f:
                res.append(json.load(f))
    
        return res
    
    def test_finalize():
        video_auth = get_auth(test_objects['vod'])
        image_auth = get_auth(test_objects['assets'])
        legacy_vod_auth = get_auth(test_objects['legacy_vod'])

        video_write = get_write_token(test_objects['vod'])
        image_write = get_write_token(test_objects['assets'])
        legacy_vod_write = get_write_token(test_objects['legacy_vod'])

        finalize_url = f"http://localhost:{port}/{test_objects['vod']}/finalize?write_token={video_write}&authorization={video_auth}"
        with timeit("Finalizing video"):
            response = requests.post(finalize_url)
        assert response.status_code == 200, response.text

        finalize_url = f"http://localhost:{port}/{test_objects['assets']}/finalize?write_token={image_write}&authorization={image_auth}"
        with timeit("Finalizing assets"):
            response = requests.post(finalize_url)
        assert response.status_code == 200, response.text
        
        finalize_url = f"http://localhost:{port}/{test_objects['legacy_vod']}/finalize?write_token={legacy_vod_write}&authorization={legacy_vod_auth}"
        with timeit("Finalizing legacy video"):
            response = requests.post(finalize_url)
        assert response.status_code == 200, response.text

        res = []
        client = ElvClient.from_configuration_url(config["fabric"]["config_url"], static_token=video_auth)
        files = client.list_files(write_token=video_write, path='video_tags/video/dummy_gpu')
        res.append(files)
        files = client.list_files(write_token=video_write, path='video_tags/video/dummy_cpu')
        res.append(files)

        client = ElvClient.from_configuration_url(config["fabric"]["config_url"], static_token=image_auth)
        files = client.list_files(write_token=image_write, path='image_tags/dummy_cpu')
        res.append(files)
        files = client.list_files(write_token=image_write, path='image_tags/dummy_gpu')
        res.append(files)
        
        client = ElvClient.from_configuration_url(config["fabric"]["config_url"], static_token=legacy_vod_auth)
        files = client.list_files(write_token=legacy_vod_write, path='video_tags/video/dummy_gpu')
        res.append(files)

        return res
    
    def test_write_token_tag():
        assets_auth = get_auth(test_objects['assets'])

        image_write = get_write_token(test_objects['assets'])
        
        image_status_url = f"http://localhost:{port}/{image_write}/status?authorization={assets_auth}"
        res = []
        tag_url = f"http://localhost:{port}/{image_write}/image_tag?authorization={assets_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_gpu": {"model":{"tags":["hello changed"]}}}, "assets": ["assets/20521092.jpg", "assets/20820751.jpg", "assets/20979342.jpg", "assets/21777769.jpg"], "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(1)
        response = requests.get(image_status_url)
        res.append(postprocess_response(response.json()))

        tag_url = f"http://localhost:{port}/{image_write}/image_tag?authorization={assets_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_cpu": {"model":{"tags":["Should not be changed"]}}}, "assets": ["assets/20521092.jpg", "assets/20820751.jpg", "assets/20979342.jpg", "assets/21777769.jpg"], "replace": False})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(10)
        response = requests.get(image_status_url)
        res.append(postprocess_response(response.json()))

        logger.debug("Waiting for server to finish tagging")
        time.sleep(75)

        response = requests.get(image_status_url).json()
        for stream in response:
            for model in response[stream]:
                assert response[stream][model]['status'] == 'Completed', response
        res.append(postprocess_response(response))

        image_tags_path = os.path.join(config["storage"]["tags"], image_write, "image")
        for tag in sorted(os.listdir(os.path.join(image_tags_path, "dummy_gpu"))):
            with open(os.path.join(image_tags_path, "dummy_gpu", tag)) as f:
                res.append(json.load(f))
                
        assert not os.path.exists(os.path.join(image_tags_path, "dummy_cpu")) or len(os.listdir(os.path.join(image_tags_path, "dummy_cpu"))) == 0
    
        return res
    
    return [test_tag, test_finalize, test_write_token_tag]

def main():
    filedir = os.path.dirname(os.path.abspath(__file__))
    tester = Tester(os.path.join(filedir, 'test_data'))
    
    def server_test():
        return test_server(args.port)

    tester.register(server_test)

    if args.record:
        tester.record(args.tests)
    else:
        tester.validate(args.tests)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--tests', nargs='+', type=str, default=None, help='Tests to run')
    parser.add_argument('--port', type=int, default=8088, help='Port to run the server on')
    args = parser.parse_args()
    main()