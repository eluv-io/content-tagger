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
        'elv-client-py @ git+https://github.com/eluv-io/elv-client-py.git@nick#egg=elv-client-py',
        'quick_test_py @ git+https://github.com/elv-nickB/quick_test_py.git#egg=quick_test_py',
        'common-ml @ git+https://github.com/elv-nickB/common-ml.git#egg=common-ml'
    ]
)