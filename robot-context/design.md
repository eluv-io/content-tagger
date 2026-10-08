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
3. `ArgsResolver` resolves request args into `TagArgs`. Per-model tenant defaults (`TenantDefaultsResolver`: tenant id `iten..` -> `iq__..` -> `public/ml_config` -> `public/profiles/model:<model>/<profile>`, falling back per model to the global profiles object `GLOBAL_PROFILES_QID`; profile is currently always `default`) are merged underneath the request; explicit request values (request `options` or job `overrides`) always win. Profiles are cached per (object, token) for `DEFAULTS_TTL` (2 min) and fetched concurrently with the content lookups (live check, default audio stream), since each is a chain of ~0.5s fabric calls.
4. `ElvClient`s are built with `create_client` (`src/common/content.py`), which caches each fabric config URL for `CONFIG_TTL` seconds instead of refetching it per client.
5. `TaggerService.tag` takes the whole `StartJobsRequest` and owns resolving it with `ArgsResolver`. `POST /tag` only authorizes synchronously; `QueueService.tag` validates the request, writes one `pending` job per requested job (`JobStore.create_job`) and returns their ids, then the request is resolved on a background executor and `QueueService.release` releases each job to `queued` with its params, parents and title (`JobStore.release_job`). Parents come from `DependencyResolver`, which only reads the job store: a job waits on the jobs in its batch that produce its dependency tracks, or failing that on queued/running/pending jobs producing them. A model is active at most once at a time on a content object: the job store refuses to create a job while another for its model on the content hasn't ended (`JobConflictError`), and `QueueService.tag` returns that job as not started while the rest of the request goes ahead (only the created jobs are resolved and released). A request naming a model twice starts it once, since its second job conflicts with the first. If resolving raises, the still-pending jobs are failed. Pending jobs left behind by a restart are not cleaned up. `JobStore.cancel_job` cancels `pending`/`queued` jobs itself and moves `running` ones to `cancelling` for the `TagRunner`.
6. The `JobStore` follows the queue manager's job lifecycle (spec: `robot-context/qmanager-api-openapi.yaml`). `jobstore.base_url` selects `QueueManagerJobStore`, otherwise `FsJobStore` (tests, local dev) emulates it.
    - Status only changes through `claim`/`complete`/`cancel`: running/pending jobs complete as succeeded/failed, cancelling ones also as cancelled (a job may end on its own before its worker sees the cancel). A job is ready once all its dependencies succeeded; a failure cascades to its dependents, a cancel to its pending/queued dependents.
    - `QueueManagerJobStore` creates jobs with type `jobstore.job_type` (default `tag`; the tests use `test-tag`, so they can clear all their jobs without touching others) and the model as subtype. The queue manager must configure that type with `auth_storage = "authorization"`: it stores the submitter's `Authorization` header, from which the client takes the token workers run with. Submitting, cancelling and deleting carry the user's token; everything else is a worker call with `X-Worker-Secret`, never both, since the secret takes precedence over a token and workers can't submit, cancel or delete. Deleting a job archives it in the queue; the client treats an archived job as missing.
    - Each job is created with uid `<qid>/<model>` (`job_uid`, the queue's `resource_hash`) and `enforce_resource: active`, so the queue refuses the submit (409) while another job with the resource hasn't ended, across all tagger instances. Release sends params, dependencies and additional_info atomically. A pending job left behind by a crash blocks its model on that content until it is cancelled.
    - The `TagRunner` lists at most its free slots of queued jobs per poll and learns of cancels from the job its progress updates return (status `cancelling`), then stops the job's model on every stream of the content, like the tagger API's stop. It completes a job with whatever the tagger reported; a hard shut down fails its running jobs.
    - `FsJobStore` keeps every job in memory and writes through to its json files, so it must be the only writer to its directory.

## 2. Data/Fabric Layer
1. Keeps track of model configs: names of models, system requirements, corresponding container image name
2. Downloads data from fabric for tagging.
3. Uploads model outputs to a `Datastore`
    - A `Datastore` is the storage protocol shared by the tagstore (text tags, filesystem or REST) and
      the vectorstore (embeddings, in-memory mock or REST).
    - A vectorstore has no first class track type, so it stubs the track endpoints and each vector
      carries its own track name.
    - A vectorstore is addressed per index, so it is built from the caller's `index_qid` rather than
      configured once at startup.
    - A vectorstore is write-only: vectors and batches are written, amended and deleted but never
      listed, so the tagstore stays the system of record for what a run produced.

## 3. Tagging Layer
1. Keeps track of which gpu containers are running on
2. Keeps track of how many resources are being used. 
3. Queues jobs and processes as resources are available
4. Is only concerned with tagging files on the system, given a container instance. It doesn't 
    care what the container does
5. Containers write tag files
6. The `TagRunner` claims queued jobs up to `tag_runner.max_jobs`. Models that declare no
    resources (every requirement is 0) don't count against that cap, since they can't starve
    anything running on the system.

## Logging
- Configured in `src/common/logging/logconfig.py`; level set by `logging.level` in the config (default INFO).
- Every line is passed through `redact()`, which strips auth tokens (`authorization=`, `token=`, bare `ascsj_...`/`eyJ...` tokens).
- Tracebacks:
    - client errors (400/403/404): single line, no traceback
    - server errors: `logger.opt(exception=e)`; full traceback (`backtrace=True`, `diagnose=True`), so redaction also covers variable values
- Job context: lines logged on behalf of a job carry `job_id` (queue id), `qid` and `model`. `TaggerWorker._handle_message` resolves each message to its job and contextualizes it; job threads are started with `_spawn` so they inherit that context. Only code that loops over several jobs (stop state, upload, heartbeat) contextualizes per job.
- API requests: one line per request (method, path, status, duration); every line logged while handling a request carries `request_id`, which is echoed in the `X-Request-ID` response header. Fast successful GETs are logged at DEBUG, except `job-status`, which is not logged at all.
- Heartbeat: the `TaggerWorker` logs a summary every `tagger.heartbeat_interval` seconds (mailbox size, actor lag, containers, GPUs) plus one line per active job. A missing heartbeat means the actor thread is stuck.
- Container failures log the container-reported errors and the tail of the container's log file.
