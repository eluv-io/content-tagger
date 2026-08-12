# Fabric Tagger

This module is used to orchestrate the flow from downloading source media, running the tag container, and uploading to the tagstore.

## Responsibilities

- Start tagging jobs on a content `FabricTagger.start`
- Check the status of a job `FabricTagger.status`
- Stop a job `FabricTagger.stop`

## Dependencies
- `ContainerScheduler`: manages scheduling the lower level containers (`TagContainer`) on the system.
- `Fetcher`: downloads data from the fabric
- `Tagstore`: interface for storing tags
- `ContainerRegistry`
    - maps requested models, by string handle, to a runnable `TagContainer` instance.
    - Also stores some basic information per model like whether they depend on an audio or video stream

## Design
- All requests and state transitions go through one single actor thread so we don't need to worry about race conditions.
- Each job is handled in it's own thread which does the following:
    - Initiates media fetching
    - Submits the container (`TagContainer`) to the `ContainerScheduler` and awaits completion using a threading.Event
    - Submits an upload request at the end
    - Updates the job status as it progresses by submitting a `JobTransition` request to the actor thread.
- There is another thread which is responsible for regularly uploading new tags to the tagstore (uses threading.Timer)
    - This is nice so we have tags uploaded before the job actually finishes
    - These uploads are submitted as requests to the main actor thread so it's still serialized.
    - There is a short delay once an upload is complete before a new request is added to the queue. This means there should never
    be more than one upload request in the queue at a given time.
- Nice benefits of the single thread model:
    - Job state updates do not need to always check if the given job was stopped by the user.
    - No synchronization needed between background tagstore uploader and the main job thread which uploads when done.
    - Status responses will never be in an inconsistent state. (arguably not that important)

## Datastore uploading
- A `source` is a concept which represents some piece of a content: likely a part or an asset. (See `src/fetch/model.py:Source`)
    - It contains the source name (i.e asset path or part hash)
    - The associated local file path
    - Optionally an `offset` in the case of video parts which informs the uploader how to adjust the timestamps
- The `TagContainer` returns a `ModelOutput` type (through it's `tags` method) which contains the local file path and it's associated tags (which are timestamped with respect the media chunk not the content object, since the `TagContainer` doesn't care about the fabric)
- The tagger keeps a mapping from the source media path to the `Source` object returned by the `Fetcher`. This way, it can take the tags from the container and align their timestamps with the content object using the offset. So tags uplaoded to the tagstore contain the global timestamp not the local one. 

### Uploader
- A `ModelTag`'s payload is its `data` field, which is either text or an embedding.
- Each job owns an `Uploader`, which routes those outputs to the store that can hold them:
  text to the tagstore and vectors to the vectorstore given by the job's `index_qid`.
- Each store gets its own `UploadSession`, and therefore its own batch. An `UploadSession` only
  knows how to write tags to one store under one batch; it does not retry and does not track
  progress.
- A model that emits a vector on a job with no `index_qid` fails the job with an error naming the
  missing option.
- The tagstore is the system of record for a run's report: every run reports there, so the
  `SourceResolver` and the status API only ever have to look in one place. The index gets a copy of
  the report when it holds vectors from the run, which means a vector-only run leaves a batch in
  the tagstore carrying the report but no tags.
- Retry (live jobs) and progress are job-level, so they live on the `Uploader`, not the sessions.
  A source is only recorded as uploaded once every store in play has taken the run's output, so a
  partial failure cannot let a diff-based re-run skip it.
- A failure in one store skips the other store's write for that tick. Nothing is re-posted on the
  next tick: a session's dedup and deletion state is only advanced after a successful write, and
  the container hands over each tag exactly once.


## Files

- All events are represented by a `Request` type, all types can be found in `message_types.py`
- Structs used to manage the state of tag jobs during their runtime go in `job_state.py`
- `model.py` defines static structs defining requests/responses to and from the fabric tagger