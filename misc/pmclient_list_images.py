### simple script to list images from podman client
### easy way to make sure podman client is working
from podman import PodmanClient

with PodmanClient() as podman_client:
    images =  podman_client.images.list()
    for image in images:
        print(image)
