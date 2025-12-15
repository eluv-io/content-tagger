FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN conda create -n mlpod python=3.10 -y
RUN apt-get update && apt-get install -y build-essential && apt-get install -y ffmpeg

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

RUN mkdir src
COPY setup.py .

RUN --mount=type=ssh conda run -n mlpod /opt/conda/envs/mlpod/bin/pip install .

COPY src ./src
COPY server.py app_config.py config.yml .

COPY version/buildinfo.py version/buildinfo.py

## the BUILD_DATE is not a secret.  but by passing it in as a secret, it is not used
## in the cache hash calculation, as such, if it changes, it does not trigger a rebuild
## HOWEVER -- when the image is rebuilt, the build date will be correctly captured

RUN --mount=type=secret,id=BUILD_DATE \
    echo "BUILD_DATE = '$(cat /run/secrets/BUILD_DATE)'" >>version/buildinfo.py

## remove this once the service actually starts using it
RUN cat version/buildinfo.py

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "server.py"]
