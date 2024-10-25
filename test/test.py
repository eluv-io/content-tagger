from quick_test_py import Tester
from argparse import ArgumentParser
import os
import shutil
from elv_client_py import ElvClient

from src.fetch_stream import fetch_stream

test_config = {
    'config_url': 'http://192.168.96.203/config?self&qspace=main',
    'mezz_qid': 'iq__42WgpoYgLTyyn4MSTejY3Y4uj81o',
    'auth_env_var': 'TEST_AUTH'
}

def get_directory_size(path: str) -> int:
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def test_fetch():
    filedir = os.path.dirname(os.path.abspath(__file__))
    client = ElvClient.from_configuration_url(config_url=test_config['config_url'], static_token=os.getenv(test_config['auth_env_var']))
    def test_audio():
        save_path = os.path.join(filedir, 'test_audio')
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        fetch_stream(content_id=test_config['mezz_qid'], start_time=0, end_time=10, stream_name="audio", client=client, output_path=save_path)
        assert os.path.exists(save_path)
        assert len(os.listdir(save_path)) == 1
        assert get_directory_size(save_path) > 1e5, get_directory_size(save_path)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        fetch_stream(content_id=test_config['mezz_qid'], stream_name="audio", client=client, output_path=save_path)
        assert os.path.exists(save_path)
        assert len(os.listdir(save_path)) > 1
        assert get_directory_size(save_path) > 1e5, get_directory_size(save_path)
        return "Passed"
    def test_video():
        save_path = os.path.join(filedir, 'test_video')
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        fetch_stream(content_id=test_config['mezz_qid'], start_time=0, end_time=10, stream_name="video", client=client, output_path=save_path)
        assert os.path.exists(save_path)
        assert len(os.listdir(save_path)) == 1
        assert get_directory_size(save_path) > 1e6
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        fetch_stream(content_id=test_config['mezz_qid'], stream_name="video", client=client, output_path=save_path)
        assert os.path.exists(save_path)
        assert len(os.listdir(save_path)) > 1
        assert get_directory_size(save_path) > 1e6
        return "Passed"
    return [test_audio, test_video]

def main():
    filedir = os.path.dirname(os.path.abspath(__file__))
    tester = Tester(os.path.join(filedir, 'test_data'))

    tester.register('test_fetch', test_fetch())

    if args.record:
        tester.record(args.tests)
    else:
        tester.validate(args.tests)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--tests', nargs='+', default=None, help='Tests to run')
    args = parser.parse_args()
    main()