### simple script to list images from podman client
### easy way to make sure podman client is working
from podman import PodmanClient


## this shows that we get the LABELS but not the attributes

with PodmanClient() as podman_client:
    images =  podman_client.images.list()
    for image in images:
        print(image, image.id)
        ## print("   ", image.attrs)
        if image.labels.get("org.opencontainers.image.vendor", "") == "Eluvio, Inc.":
            for key, value in image.labels.items():
                print(f"      {key}: {value}")
        else:
            print("      Not Eluvio, Inc.")

