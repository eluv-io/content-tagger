import pytest

from src.common.model import ModelConfig
from src.service.job_poster import JobPoster
from src.tagging.fabric_tagging.queue.model import ListJobArgs

@pytest.fixture
def model_configs():
    return {
        "model1": ModelConfig(
            image="not important",
            description="not important",
            type="video",
            resources={},
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
            track_dependencies=["model1", "some-track1", "some-track2"],
        ),
        "not-runnable": ModelConfig(
            image="not important",
            description="not important",
            type="video",
            resources={},
            track_dependencies=["doesn't exist"],
        ),
    }

@pytest.fixture
def job_poster(queue_jobstore, track_resolver, fake_qapifactory, model_configs) -> JobPoster:
    """Create a JobPoster for testing, using the queue_jobstore and other dependencies."""
    return JobPoster(
        job_store=queue_jobstore,
        track_resolver=track_resolver,
        model_configs=model_configs,
        qfactory=fake_qapifactory
    )


def test_queued_dependencies_separate_requests(q, job_poster: JobPoster, make_tag_args):
    jobstore = job_poster.jobstore
    res = job_poster.post_jobs(q, [make_tag_args(feature="model1")])
    job_id1 = res[0].job_id

    # check that a second submission is rejected
    res = job_poster.post_jobs(q, [make_tag_args(feature="model1")])
    assert res[0].started is False

    # check that posting model2, will create a dependency
    res = job_poster.post_jobs(q, [make_tag_args(feature="model2")])
    job_id2 = res[0].job_id
    assert len(res[0].dependencies) == 1 and res[0].dependencies[0] == job_id1

    # extend the chain
    res = job_poster.post_jobs(q, [make_tag_args(feature="model3")])
    job_id3 = res[0].job_id
    assert set(res[0].dependencies) == {job_id1, job_id2}

    # check that the dependencies exist in the jobstore
    assert jobstore.get_job(job_id1).deps == []
    assert jobstore.get_job(job_id2).deps == [job_id1]
    assert set(jobstore.get_job(job_id3).deps) == {job_id1, job_id2}

def test_queued_dependencies_same_request(q, job_poster: JobPoster, make_tag_args):
    jobstore = job_poster.jobstore
    res = job_poster.post_jobs(q, [make_tag_args(feature="model1")])
    job_id1 = res[0].job_id

    # check that posting model2, will create a dependency
    res = job_poster.post_jobs(q, [make_tag_args(feature="model2"), make_tag_args(feature="model1")])
    assert len(res[0].dependencies) == 1 and res[0].dependencies[0] == job_id1

    # check second was rejected (already running)
    assert res[1].started is False

def test_mixed_dependencies(q, job_poster: JobPoster, make_tag_args):
    res = job_poster.post_jobs(q, [make_tag_args(feature="model1")])
    job_id1 = res[0].job_id

    # post model2 and model3 together
    res = job_poster.post_jobs(q, [make_tag_args(feature="model1"), make_tag_args(feature="model2"), make_tag_args(feature="model3")])
    assert res[0].started is False
    assert res[1].started is True
    assert res[2].started is True

    assert len(res[1].dependencies) == 1 and res[1].dependencies[0] == job_id1
    job_id2 = res[1].job_id
    assert set(res[2].dependencies) == {job_id1, job_id2}

def test_missing_dependency_runs_anyway(q, job_poster: JobPoster, make_tag_args):
    jobstore = job_poster.jobstore
    res = job_poster.post_jobs(q, [make_tag_args(feature="model2")])
    job_id1 = len(res[0].dependencies) == 0

    # make sure it's claimable by worker
    assert len(jobstore.list_jobs(ListJobArgs(), "test-auth")) == 1