import pytest

from src.common.model import ModelConfig
from src.service.impl.queue_based import QueueService
from src.service.dependency_resolver import DependencyResolver
from src.tagging.fabric_tagging.queue.model import CreateQueueItem, ListJobArgs
from tests.core_tagging.conftest import enqueue

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

def test_queued_dependencies_separate_requests(q, queue_client: QueueService, make_tag_args):
    jobstore = queue_client.jobstore
    job_id1 = enqueue(queue_client, q, [make_tag_args(feature="model1")])[0]

    # check that a second submission is cancelled as a duplicate
    dup_id = enqueue(queue_client, q, [make_tag_args(feature="model1")])[0]
    assert jobstore.get_job(dup_id).status == "cancelled"

    # check that posting model2, will create a dependency
    job_id2 = enqueue(queue_client, q, [make_tag_args(feature="model2")])[0]

    # extend the chain
    job_id3 = enqueue(queue_client, q, [make_tag_args(feature="model3")])[0]

    # check that the dependencies exist in the jobstore
    assert jobstore.get_job(job_id1).deps == []
    assert jobstore.get_job(job_id2).deps == [job_id1]
    assert set(jobstore.get_job(job_id3).deps) == {job_id1, job_id2}

def test_queued_dependencies_same_request(q, queue_client: QueueService, make_tag_args):
    jobstore = queue_client.jobstore
    job_id1 = enqueue(queue_client, q, [make_tag_args(feature="model1")])[0]

    # check that posting model2, will create a dependency
    ids = enqueue(queue_client, q, [make_tag_args(feature="model2"), make_tag_args(feature="model1")])
    assert jobstore.get_job(ids[0]).deps == [job_id1]

    # check second was cancelled (already running)
    assert jobstore.get_job(ids[1]).status == "cancelled"

def test_mixed_dependencies(q, queue_client: QueueService, make_tag_args):
    jobstore = queue_client.jobstore
    job_id1 = enqueue(queue_client, q, [make_tag_args(feature="model1")])[0]

    # post model2 and model3 together
    ids = enqueue(queue_client, q, [make_tag_args(feature="model1"), make_tag_args(feature="model2"), make_tag_args(feature="model3")])
    assert jobstore.get_job(ids[0]).status == "cancelled"
    assert jobstore.get_job(ids[1]).status == "queued"
    assert jobstore.get_job(ids[2]).status == "queued"

    assert jobstore.get_job(ids[1]).deps == [job_id1]
    assert set(jobstore.get_job(ids[2]).deps) == {job_id1, ids[1]}

def test_mixed_dependencies2(q, queue_client: QueueService, make_tag_args):
    jobstore = queue_client.jobstore
    job_id1 = enqueue(queue_client, q, [make_tag_args(feature="model1")])[0]
    job_id2 = enqueue(queue_client, q, [make_tag_args(feature="model2")])[0]

    # post model2 and 3 in the same request, and make sure that model3 depends on the previously submitted job
    ids = enqueue(queue_client, q, [make_tag_args(feature="model2"), make_tag_args(feature="model3")])
    assert jobstore.get_job(ids[0]).status == "cancelled"
    assert jobstore.get_job(ids[1]).status == "queued"
    assert set(jobstore.get_job(ids[1]).deps) == {job_id1, job_id2}

def test_duplicate_within_request(q, dependency_resolver: DependencyResolver, make_tag_args):
    deps = dependency_resolver.resolve(q, ["a", "b", "c"], [make_tag_args(feature="model1"), make_tag_args(feature="model1"), make_tag_args(feature="model2")])
    assert deps[0].duplicate_of is None
    assert deps[1].duplicate_of == "a"
    assert deps[2].parents == ["a"]

def test_waits_on_pending_job(q, queue_client: QueueService, make_tag_args):
    pending = queue_client.jobstore.create_job(CreateQueueItem(qid=q.qid, model="model1"), auth=q.token)
    deps = queue_client.dependency_resolver.resolve(q, ["x"], [make_tag_args(feature="model2")])
    assert deps[0].parents == [pending.id]

def test_missing_dependency_runs_anyway(q, queue_client: QueueService, make_tag_args):
    jobstore = queue_client.jobstore
    job_id = enqueue(queue_client, q, [make_tag_args(feature="model2")])[0]
    assert jobstore.get_job(job_id).deps == []

    # make sure it's claimable by worker
    assert len(jobstore.list_jobs(ListJobArgs(), "test-auth")) == 1
