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

test_config = {
    'test_video': 'iq__3C58dDYxsn5KKSWGYrfYr44ykJRm',
    'test_assets': 'iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2',
    'video_auth': 'VIDEO_AUTH',
    'video_write_token': 'VIDEO_WRITE',
    'assets_auth': 'ASSETS_AUTH',
    'assets_write_token': 'ASSETS_WRITE',
    'available_gpus': 5
}

def postprocess_response(res: dict):
    for stream in res:
        for model in res[stream]:
            del res[stream][model]['tag_job_id']
            del res[stream][model]['time_running']
    return res

def test_server(port: int) -> List[Callable]:
    filedir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Starting server on port {args.port}")
    server_out = open(os.path.join(filedir, 'server_out.log'), 'w')
    process = subprocess.Popen(['python', os.path.join(filedir, "../server.py"), '--port', str(args.port)], stdout=server_out, stderr=server_out)
    time.sleep(2)
    def cleanup():
        logger.info("Cleaning up")
        server_out.close()
        process.terminate()

    atexit.register(cleanup)

    assets_auth = os.getenv(test_config['assets_auth'])
    video_auth = os.getenv(test_config['video_auth'])
    assert assets_auth is not None, "Please set the ASSETS_AUTH environment variable"
    assert video_auth is not None, "Please set the VIDEO_AUTH environment variable"
    
    parts_path = os.path.join(config["storage"]["parts"], test_config['test_video'])
    image_path = os.path.join(config["storage"]["images"], test_config['test_assets'])
    
    shutil.rmtree(parts_path, ignore_errors=True)
    shutil.rmtree(image_path, ignore_errors=True)

    # test cases
    def test_tag():
        res = []
        video_status_url = f"http://localhost:{port}/{test_config['test_video']}/status?authorization={video_auth}"
        response = requests.get(video_status_url)
        assert response.status_code == 404, response.text
        res.append(response.json())

        tag_url = f"http://localhost:{port}/{test_config['test_video']}/tag?authorization={video_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_gpu": {"model":{"tags":["hello1", "hello2"]}}}, "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(1)
        response = requests.get(video_status_url)
        res.append(postprocess_response(response.json()))

        tag_url = f"http://localhost:{port}/{test_config['test_video']}/tag?authorization={video_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_cpu": {"model":{"tags":["a", "b", "a"], "allow_single_frame":False}}}, "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(10)

        response = requests.get(video_status_url)
        res.append(postprocess_response(response.json()))

        image_status_url = f"http://localhost:{port}/{test_config['test_assets']}/status?authorization={assets_auth}"

        tag_url = f"http://localhost:{port}/{test_config['test_assets']}/image_tag?authorization={assets_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_gpu": {"model":{"tags":["hello1"]}}}, "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(1)
        response = requests.get(image_status_url)
        res.append(postprocess_response(response.json()))

        tag_url = f"http://localhost:{port}/{test_config['test_assets']}/image_tag?authorization={assets_auth}"
        response = requests.post(tag_url, json={"features": {"dummy_cpu": {"model":{"tags":["hello2"]}}}, "replace": True})
        assert response.status_code == 200, response.text
        res.append(response.json())
        time.sleep(10)
        response = requests.get(image_status_url)
        res.append(postprocess_response(response.json()))

        logger.debug("Waiting for server to finish tagging")
        time.sleep(75)

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

        video_tags_path = os.path.join(config["storage"]["tags"], test_config['test_video'], "video")
        for tag in sorted(os.listdir(os.path.join(video_tags_path, "dummy_gpu"))):
            with open(os.path.join(video_tags_path, "dummy_gpu", tag)) as f:
                res.append(json.load(f))
        for tag in sorted(os.listdir(os.path.join(video_tags_path, "dummy_cpu"))):
            with open(os.path.join(video_tags_path, "dummy_cpu", tag)) as f:
                res.append(json.load(f))

        image_tags_path = os.path.join(config["storage"]["tags"], test_config['test_assets'], "image")
        for tag in sorted(os.listdir(os.path.join(image_tags_path, "dummy_gpu"))):
            with open(os.path.join(image_tags_path, "dummy_gpu", tag)) as f:
                res.append(json.load(f))
        for tag in sorted(os.listdir(os.path.join(image_tags_path, "dummy_cpu"))):
            with open(os.path.join(image_tags_path, "dummy_cpu", tag)) as f:
                res.append(json.load(f))
    
        return res
    
    def test_finalize():
        video_auth = os.getenv(test_config['video_auth'])
        image_auth = os.getenv(test_config['assets_auth'])

        video_write = os.getenv(test_config['video_write_token'])
        image_write = os.getenv(test_config['assets_write_token'])

        finalize_url = f"http://localhost:{port}/{test_config['test_video']}/finalize?write_token={video_write}&authorization={video_auth}"
        with timeit("Finalizing video"):
            response = requests.post(finalize_url)
        assert response.status_code == 200, response.text

        finalize_url = f"http://localhost:{port}/{test_config['test_assets']}/finalize?write_token={image_write}&authorization={image_auth}"
        with timeit("Finalizing assets"):
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

        return res
    
    return [test_tag, test_finalize]

def main():
    filedir = os.path.dirname(os.path.abspath(__file__))
    tester = Tester(os.path.join(filedir, 'test_data'))

    all_tests = {"server_test": lambda: test_server(args.port)}

    for testname, test in all_tests.items():
        if args.tests and testname not in args.tests:
            continue
        else:
            tester.register(testname, test())

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