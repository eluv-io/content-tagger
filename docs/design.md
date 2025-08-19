# Tagger

## 1. API Layer
1. Flask handlers
2. formatting args and responses

## 2. Data/Fabric Layer
1. Keeps track of model configs: names of models, system requirements, corresponding container image name
2. Downloads data from fabric for tagging.
3. Uploads tags to TagsDB (or fabric for legacy)

## 3. Tagging Layer
1. Keeps track of which gpu containers are running on
2. Keeps track of how many resources are being used. 
3. Queues jobs and processes as resources are available
4. Is only concerned with tagging files on the system, given a container instance. It doesn't 
    care what the container does
5. Containers write tag files