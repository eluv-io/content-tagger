# Eluvio Tagger

The Eluvio Tagger is a service for running ML tagging models against content stored in the Content Fabric. It orchestrates pluggable model containers, manages a job queue with automatic dependency resolution, and writes the resulting tags back to a tagstore aligned to the source media's timeline.

## API docs

- Tagger job management: 
    - html: https://ai.contentfabric.io/tagging-live/docs
    - swagger: https://ai.contentfabric.io/tagging-live/openapi.json
- Tagstore (viewing existing tags): 
    - html: https://ai.contentfabric.io/tagstore/docs


## Core Concepts

### Tagging by content

Tagging is performed against a **content id** (`qid`) — the identifier that uniquely addresses a content object in the Fabric. Every tagging request and job is scoped to a single `qid`.

### Tagging Scopes

A **scope** selects the facet of the content you want to tag and controls how the media is broken into chunks for processing:

| Scope | Description |
|-------|-------------|
| **video** | Standard VOD video/audio tagging. Generates tags by **part** (~30s). The media stream is configurable when tagging with video scope.|
| **assets** | Tags static image assets attached as files to the content. |
| **livestream** | Tags segments of a livestream. |
| **tag-aligned** | Chunks the media based on the start/end times of another tag track, or alternatively into fixed equal-size segments (e.g. `5s`). |

The scope lets you target exactly the portion and granularity of content that makes sense for a given model.

#### Starting before the stream

Content is treated as live if its playout options (`/rep/playout/options.json`) report `live: true`, even when the stream isn't running yet. A live job can therefore be started ahead of time: the tagging container boots immediately so the model is loaded and ready, and fetching polls until the stream is running (it has an edge write token). If the stream hasn't started within `start_timeout` seconds (livestream scope, default 3600) the job fails.

#### Live job status

A livestream has no defined end, so a stream that ends or is stopped surfaces to the tagger as a fetch or container error — indistinguishable from a genuine failure. Because of that, a live job that errors is reported with status **`cancelled`** rather than `failed`, matching what a manual stop would produce. The underlying error is still returned in the `error` field, and the job's own record keeps its `failed` state, so no diagnostic information is lost.

### Tagstore

The tagstore is a lightweight storage layer that sits in front of the content fabric. Tags can be written back to the content fabric as needed to take advantage of the guarantees and versioning of the content fabric. 

Tags are written to the **tagstore**. Model containers emit tags relative to the chunk of media they were given; the tagger is responsible for aligning those timestamps back against the full content's timeline.

##### Tag Tracks

Tags are grouped by **track**. A model declares the track(s) it produces.

### Diff-based tagging

Tagging can be run with `replace=true` or `replace=false`:

- **`replace=false`** (default) — diff-based. The tagger only tags parts of the media that have not already been tagged, skipping previously-tagged sources.
- **`replace=true`** — new tags **shadow** the tags produced by previous tagger jobs, re-tagging the content with a fresh pass.

On **live** content, `replace=true` additionally deletes every existing tag from the same model before the job starts (the same operation as `DELETE /{qid}/tags/{model}`). Live segment indexes restart with the stream, so leaving old tags in place would collide with the new run. 

### Tenant defaults

Per-model default job parameters come from named profiles. The tagger resolves the tenant of the content being tagged, reads the ml config object id from the tenant object's `public/ml_config`, and loads `public/profiles` from it. Each `model:<model>` key maps profile names to job parameters (`model_params`, `track_suffix`, `overrides`, ...), in the same format as a job in the tag request. If the tenant doesn't define the profile for a model, the same lookup is done on the global profiles object (`GLOBAL_PROFILES_QID` in `src/api/tenant_defaults.py`). Currently the `default` profile is always used. `caller_info` cannot be set by tenant defaults.

These defaults are merged underneath the request. Anything set explicitly in the request takes precedence, including request-level `options` over a tenant default's `overrides`. Precedence from lowest to highest: tenant defaults → request `options` → job `overrides`.

### Extending the tagger

Models are easily **pluggable** into the tagger runtime. Each model is an OCI container that implements a standardized communication protocol, so adding a new capability is a matter of building a conforming container. See the protocol documentation for the details of how containers receive input and emit tags as well as how to easily build new containers.

### Model Categories

Every model declares a **category** describing what it is used for. The `/models` endpoint returns it
alongside the model, so clients can group the listing. An empty category means unset:

| Category | Description |
|----------|-------------|
| **Description & Transcription** | Semantically rich, free-form descriptions of the video, useful for search and summary (e.g. `asr`, `llava`, `scene_description`). |
| **Frame Level Detection** | Detects entities in the video, with short categorical outputs rather than natural language (e.g. `ocr`, `celeb`, `logo`, `speaker`, `caption`). |
| **Segmentation** | Tags used mainly for their start/end times rather than the tag itself (e.g. `shot`). |
| **Special** | Specialized models run to support a specific product rather than for general video understanding (e.g. vertical video, `pose`, `chapters`, `evidence`). |

### Model Parameters

The `/models` endpoint also returns each model's `params_schema`: an OpenAPI schema describing the `model_params` it accepts, including types, defaults and required fields. It is `null` for models without documented parameters.

### Dependency Management

Some models depend on the tag outputs of other models. The **`/models`** endpoint returns the available models along with their associated dependent tracks. The tagger automatically **resolves these dependencies** and runs the models in the correct order.

The tagger does not automatically queue all dependencies it is the responsibility of the caller to decide which model to run to satisfy the track dependency. The tagger will wait on dependencies in the following two cases: 
1. The dependencies are submitted with the dependent job within the same request.
2. The dependencies have already been submitted and have not yet completed.

A job only starts once every job it waits on has `succeeded`. If one of them fails, the job and everything waiting on it fail too; if one is cancelled before it runs, they are cancelled. `error` names the parent job in both cases.

### Job lifecycle

`POST /{qid}/tag` returns as soon as the request is authorized: each job is written in the **`pending`** state and its id is returned immediately, so it can be polled with `job-status` right away. Request parameters, tenant defaults and dependencies are resolved in the background, after which the job moves to **`queued`**, then **`running`**, and ends as `succeeded`, `failed` or `cancelled`.

- A pending job has empty `params` and `stream` until it is resolved.
- If resolving fails (for example invalid model parameters after applying tenant defaults), the job goes to `failed` with the reason in `error`.
- A model runs at most once at a time on a content object, whatever the stream. A requested job whose model already has a job on the content that hasn't ended (including an earlier job in the same request) isn't created: it comes back with `started: false` and a message saying so, while the rest of the request goes ahead.
- Stopping a `pending` or `queued` job cancels it immediately; stopping a `running` job moves it to **`cancelling`** until the worker has stopped it, after which it is `cancelled`.
- A job whose worker shuts down while it is running is `failed`.


## Viewing Tags in EVIE

Tags produced by the tagger are viewable through the **EVIE** UI, where they can be browsed and inspected against the content timeline.

![Tags in EVIE](TODO-add-image-path)