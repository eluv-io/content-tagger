from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flask',
        'flask_cors',
        'loguru',
        'podman',
        'pynvml',
        'elv-client-py @ git+https://github.com/eluv-io/elv-client-py.git#egg=elv-client-py',
        'quick_test_py @ git+https://github.com/eluv-io/quick-test-py.git#egg=quick_test_py',
        'common_ml @ git+ssh://git@github.com/eluv-io/common-ml.git#egg=common_ml',
    ]
)