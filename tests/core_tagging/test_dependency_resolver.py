from unittest.mock import Mock

import pytest

from src.api.tagging.request_format import JobSpec, StartJobsRequest
from src.common.model import ModelConfig
from src.service.impl.queue_based import QueueService
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs
from tests.core_tagging.conftest import enqueue

@pytest.fixture
def released_deps(queue_client: QueueService, monkeypatch) -> dict[str, list[str]]:
    """Log the dependencies for each job - so that we can validate dependency resolution is correct
    """
    deps: dict[str, list[str]] = {}
    release = queue_client.jobstore.release_job
    def record(args, auth):
        deps[args.id] = args.deps
        return release(args, auth)
    monkeypatch.setattr(queue_client.jobstore, "release_job", record)
    return deps

@pytest.fixture
def model_configs():
    return {
        "model1": ModelConfig(
            image="not important",
            description="not important",
            type="video",
            resources={}
        ),
        "model2": ModelConfig(
            image="not important",
            description="not important",
            type="video",
            resources={},
            track_dependencies=["model1"],
            track_outputs=["some-track1", "some-track2"]
        ),
        "model3": ModelConfig(
            image="not important",
            description="not important",
            type="video",
            resources={},
            track_dependencies=["model1", "some-track1", "some-track2"]
        ),
        "not-runnable": ModelConfig(
            image="not important",
            description="not important",
            type="video",
            resources={},
            track_dependencies=["doesn't exist"]
        ),
    }

def test_queued_dependencies_separate_requests(q, queue_client: QueueService, make_tag_args, released_deps):
    job_id1 = enqueue(queue_client, q, [make_tag_args(feature="model1")])[0]
    job_id2 = enqueue(queue_client, q, [make_tag_args(feature="model2")])[0]
    job_id3 = enqueue(queue_client, q, [make_tag_args(feature="model3")])[0]

    assert released_deps[job_id1] == []
    assert released_deps[job_id2] == [job_id1]
    assert set(released_deps[job_id3]) == {job_id1, job_id2}

def test_queued_dependencies_same_request(q, queue_client: QueueService, make_tag_args, released_deps):
    ids = enqueue(queue_client, q, [make_tag_args(feature="model1"), make_tag_args(feature="model2"), make_tag_args(feature="model3")])

    assert released_deps[ids[0]] == []
    assert released_deps[ids[1]] == [ids[0]]
    assert set(released_deps[ids[2]]) == {ids[0], ids[1]}

def test_mixed_dependencies(q, queue_client: QueueService, make_tag_args, released_deps):
    job_id1 = enqueue(queue_client, q, [make_tag_args(feature="model1")])[0]

    # model3 waits on model2 from its own request and on the earlier model1
    ids = enqueue(queue_client, q, [make_tag_args(feature="model2"), make_tag_args(feature="model3")])
    assert released_deps[ids[0]] == [job_id1]
    assert set(released_deps[ids[1]]) == {job_id1, ids[0]}

def test_waits_on_pending_job(q, queue_client: QueueService, make_tag_args):
    pending = queue_client.jobstore.create_job(CreateQueueItem(qid=q.qid, model="model1"), auth=q.token)
    parents = queue_client.dependency_resolver.resolve(q, ["x"], [make_tag_args(feature="model2")])
    assert parents[0] == [pending.id]

def test_missing_dependency_runs_anyway(q, queue_client: QueueService, make_tag_args, released_deps):
    jobstore = queue_client.jobstore
    job_id = enqueue(queue_client, q, [make_tag_args(feature="model2")])[0]
    assert released_deps[job_id] == []

    # make sure it's claimable by worker
    assert len(jobstore.list_jobs(ListJobArgs(qid=q.qid), q.token)) == 1

def test_child_waits_on_existing_job_when_its_parent_in_the_request_isnt_started(q, queue_client: QueueService, make_tag_args, released_deps):
    job_id1 = enqueue(queue_client, q, [make_tag_args(feature="model1")])[0]
    queue_client.arg_resolver.resolve.side_effect = lambda req, q: [make_tag_args(feature=job.model) for job in req.jobs] # type: ignore

    model1, model2 = queue_client.tag(q, StartJobsRequest(jobs=[JobSpec(model="model1"), JobSpec(model="model2")]))

    assert not model1.started
    assert released_deps[model2.job_id] == [job_id1]
