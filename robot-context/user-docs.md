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

### Tagstore

The tagstore is a lightweight storage layer that sits in front of the content fabric. Tags can be written back to the content fabric as needed to take advantage of the guarantees and versioning of the content fabric. 

Tags are written to the **tagstore**. Model containers emit tags relative to the chunk of media they were given; the tagger is responsible for aligning those timestamps back against the full content's timeline.

##### Tag Tracks

Tags are grouped by **track**. A model declares the track(s) it produces.

### Diff-based tagging

Tagging can be run with `replace=true` or `replace=false`:

- **`replace=false`** (default) — diff-based. The tagger only tags parts of the media that have not already been tagged, skipping previously-tagged sources.
- **`replace=true`** — new tags **shadow** the tags produced by previous tagger jobs, re-tagging the content with a fresh pass. 

### Extending the tagger

Models are easily **pluggable** into the tagger runtime. Each model is an OCI container that implements a standardized communication protocol, so adding a new capability is a matter of building a conforming container. See the protocol documentation for the details of how containers receive input and emit tags as well as how to easily build new containers.

### Dependency Management

Some models depend on the tag outputs of other models. The **`/models`** endpoint returns the available models along with their associated dependent tracks. The tagger automatically **resolves these dependencies** and runs the models in the correct order.

The tagger does not automatically queue all dependencies it is the responsibility of the caller to decide which model to run to satisfy the track dependency. The tagger will wait on dependencies in the following two cases: 
1. The dependencies are submitted with the dependent job within the same request.
2. The dependencies have already been submitted and have not yet completed.


## Viewing Tags in EVIE

Tags produced by the tagger are viewable through the **EVIE** UI, where they can be browsed and inspected against the content timeline.

![Tags in EVIE](TODO-add-image-path)