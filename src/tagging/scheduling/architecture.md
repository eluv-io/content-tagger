# System Tagger

This module is used to schedule container jobs on the machine (see [TagContainer notes](../../tag_containers/architecture.md)). 

## Responsibilities

Container jobs involve running a model that lives in a container and generating output tags, but what exactly the job does is irrelevant to the ContainerScheduler which is only responsible for queuing them and pressing the tag button when they are ready. It is also responsible for reporting to the caller (via a threading.Event) when the job is done.

## Methods
The following methods are provided which are fairly intuitive
- `start`: Takes a `TagContainer` and optionally a `threading.Event` so the caller can be notified when the container has exited. Returns a jobid handle.
- `stop`: Takes a jobid handle.
- `status`: Takes a jobid handle and return a status struct

## Dependencies

- `SysConfig` object, which is used to describe the systems resources. An administrator will need to set this manually since there is not a discovery procedure currently (TODO)
- `TagContainer` instances are passed directly through the methods.

## Design

- Similar to the [FabricTagger](../fabric_tagging/architecture.md), the system tagger also uses a single thread actor model to manage concurrent requests. This helps to  prevent race conditions that might arise from many different threads reading and modifying the same job state objects (`state.py`).
- The types of messages the actor thread manages are in `message_types.py`
- There is a background thread which runs simply to check all active containers whether they have exited or not. If they have then it submits a message to the main thread to update the jobs status to finished.