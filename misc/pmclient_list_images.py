### simple script to list images from podman client
### easy way to make sure podman client is working
from podman import PodmanClient
import json
import time

## this shows that we get the LABELS but not the attributes

with PodmanClient() as podman_client:
    images =  podman_client.images.list()
    for image in images:
        print("-----------", image) 
        ##print("   ", json.dumps(image.attrs, indent=2))
        if image.labels.get("org.opencontainers.image.vendor", "") == "Eluvio, Inc.":
            for key, value in image.labels.items():
                print(f"      {key}: {value}")
            print(    f"      {image.attrs.get('Created', 0)}")
            print(    f"      {time.ctime(image.attrs.get('Created', 0))}")
        else:
            print("      Not Eluvio, Inc.")

