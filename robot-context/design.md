# Tagger

## Fabric Tagger
### Purpose
Manage the full workflow of a tagging job: 1. Fetching parts/data, 2. Scheduling the tag work to be done, 3. Awaiting step 2 and uploading remaining tags
### Design
- Uses a single-threaded actor model to modify job state and transition from active to inactive job. 
- Uses several worker threads to supervise the tagging workflow, each of these calls the actor thread to update job state.
- Provides external facing API: status/stop/start/cleanup
### Actor responsibilities
- Start tagging job (user)
- Check job status (user)
- Stop job (user or worker thread)
    - Worker thread may also stop the job in the case of errors or completion
- Cleanup: stops all jobs (application)
- Transition job state (worker thread only)
    - Starting -> Fetching -> Tagging -> Uploading tags -> Complete
    - This only updates the job state struct, the worker thread is responsible for doing the actual work. 
- Upload (background thread)
    - Regularly checks running jobs and uploads any new tags.
    - Right now this upload is scheduled via the actor, but it can probably be made asynchronous for performance. 

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