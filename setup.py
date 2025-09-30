from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'marshmallow==3.26.1',
        'flask',
        'flask_cors',
        'loguru',
        'podman',
        'pynvml',
        'setproctitle',
        'waitress==3.0.0',
        'psutil',
        'dacite',
        'elv-client-py @ git+https://github.com/eluv-io/elv-client-py.git#egg=elv-client-py',
        'common_ml @ git+ssh://git@github.com/eluv-io/common-ml.git#egg=common_ml',
    ]
)
