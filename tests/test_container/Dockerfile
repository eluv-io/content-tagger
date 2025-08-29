FROM continuumio/miniconda3:latest

WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential \
    && apt-get install -y ffmpeg

RUN \
   conda create -n dummy python=3.10 -y

SHELL ["conda", "run", "-n", "dummy", "/bin/bash", "-c"]

COPY . .

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

RUN /opt/conda/envs/dummy/bin/pip install -r requirements.txt

ENTRYPOINT ["/opt/conda/envs/dummy/bin/python", "run.py"]