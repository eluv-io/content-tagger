FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN conda create -n mlpod python=3.10 -y

SHELL ["conda", "run", "-n", "mlpod", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y build-essential && apt-get install -y ffmpeg

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

RUN mkdir src
COPY setup.py .

RUN /opt/conda/envs/mlpod/bin/pip install .

COPY src ./src
COPY server.py app_config.py config.yml .

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "server.py"]
