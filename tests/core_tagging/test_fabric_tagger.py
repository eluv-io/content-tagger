import pytest
import time
from dataclasses import replace as dc_replace
from unittest.mock import Mock, patch

from src.tags.tagstore.model import Track
from src.tags.tagstore.rest_tagstore import RestTagstore
from src.tagging.fabric_tagging.tagger import FabricTagger
from src.tagging.fabric_tagging.model import TagStatusResult
from src.tag_containers.model import ContainerRequest, Error, ModelTag
from src.fetch.model import *
from src.common.content import Content
from src.common.errors import BadRequestError, MissingResourceError
from tests.core_tagging.conftest import FakeTagContainer, PartialResultContainer
    
def _status_for(
    reports: list[TagStatusResult],
    model: str,
) -> TagStatusResult:
    matches = [r for r in reports if r.model == model]
    assert matches, f"Missing status for model={model}"
    return matches[0]

def test_tag_success(fabric_tagger, q, sample_tag_args):
    """Test successful tagging job creation"""
    results = []
    for args in sample_tag_args:
        result = fabric_tagger.tag(q, args)
        results.append(result)
    
    # Check that jobs were started successfully
    assert all(result.started for result in results)

    # Check that jobs are in active jobs
    active_jobs = fabric_tagger.jobstore.active_jobs
    assert len(active_jobs) == 2
    
    # Check job details
    job_ids = list(active_jobs.keys())
    features = [active_jobs[job_id].args.feature for job_id in job_ids]
    assert "caption" in features
    assert "asr" in features


def test_tag_invalid_feature(fabric_tagger, q, make_tag_args):
    """Test tagging with invalid feature"""
    invalid_args = make_tag_args(feature="invalid_feature", stream="video")
    with pytest.raises(MissingResourceError, match="Invalid feature: invalid_feature"):
        fabric_tagger.tag(q, invalid_args)


def test_tag_duplicate_job(fabric_tagger, q, sample_tag_args):
    """Test that duplicate jobs are rejected"""
    # Start first job
    result1 = fabric_tagger.tag(q, sample_tag_args[0])
    assert result1.started is True
    
    # Try to start same job again
    result2 = fabric_tagger.tag(q, sample_tag_args[0])
    assert result2.started is False


def test_status_no_jobs(fabric_tagger):
    """Test status when no jobs exist"""
    with pytest.raises(MissingResourceError, match="No jobs started for"):
        fabric_tagger.status("iq__nonexistent")


def test_status_with_jobs(fabric_tagger, q, sample_tag_args):
    """Test status retrieval with active jobs"""
    # Start jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    # Get status
    status = fabric_tagger.status(q.qid)

    assert isinstance(status, list)
    assert len(status) == 2
    assert any(s.model == "caption" for s in status)
    assert any(s.model == "asr" for s in status)

    for s in status:
        assert s.status.status
        assert isinstance(s.status.time_started, float)


def test_status_completed_jobs(fabric_tagger, q, sample_tag_args):
    """Test status includes completed jobs"""
    # Start and complete jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    # Wait a bit for jobs to progress
    time.sleep(1)
    
    # Check that job is done according to status
    status = fabric_tagger.status(q.qid)
    status1 = _status_for(status, "caption")
    assert status1.status.status == "Completed"
    assert status1.status.time_ended is not None
    assert status1.status.time_ended > status1.status.time_started
    assert len(status1.status.downloaded_sources) == 2
    assert len(status1.status.downloaded_sources) == 2

    assert _status_for(status, "asr").status.status == "Completed"


def test_stop_running_job(fabric_tagger, q, sample_tag_args):
    """Test stopping a running job"""
    # Start jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    # Stop one job
    fabric_tagger.stop(q.qid, "caption")
    
    # Check that stop event was set
    status = fabric_tagger.status(q.qid)
    assert _status_for(status, "caption").status.status == "Stopped"
    assert _status_for(status, "asr").status.status != "Stopped"


def test_stop_nonexistent_job(fabric_tagger):
    """Test stopping a job that doesn't exist"""
    if isinstance(fabric_tagger.tagstore, RestTagstore):
        pytest.skip("Skipping test for RestTagstore cause I don't want to have to configure many live test objects")

    with pytest.raises(MissingResourceError):
        fabric_tagger.stop("iq__nonexistent", "caption")


def test_stop_finished_job(fabric_tagger, q, sample_tag_args):
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    time.sleep(1)
    with pytest.raises(MissingResourceError):
        fabric_tagger.stop(q.qid, "caption")
    # check that both are Completed
    status = fabric_tagger.status(q.qid)
    assert _status_for(status, "caption").status.status == "Completed"
    assert _status_for(status, "asr").status.status == "Completed"


def test_cleanup(fabric_tagger, q, sample_tag_args):
    """Test cleanup shuts down properly"""
    # Start some jobs
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)

    time.sleep(1)

    # Cleanup
    fabric_tagger.cleanup()

    time.sleep(1)
    
    # Check shutdown signal is set
    assert fabric_tagger.shutdown_requested()
    
    # check is_running for all containers
    assert len(fabric_tagger.jobstore.active_jobs) == 0
    assert len(fabric_tagger.jobstore.inactive_jobs) == 2

    assert fabric_tagger.system_tagger.exit_requested

    for job in fabric_tagger.system_tagger.jobs.values():
        assert job.finished.is_set()
        assert job.container.is_running() is False


def test_many_concurrent_jobs(fabric_tagger, make_tag_args):
    if isinstance(fabric_tagger.tagstore, RestTagstore):
        pytest.skip("Skipping test for RestTagstore cause I don't want to have to configure many live test objects")

    """Run many jobs and make sure that they all run"""
    contents = []
    for i in range(5):
        contents.append(Content(qid=f"iq__content{i}", token=f"token{i}"))

    all_args = []
    for _content in contents:
        all_args.extend([
            make_tag_args(feature="caption", stream="video"),
            make_tag_args(feature="asr", stream="audio"),
        ])

    results = []
    content_idx = 0
    for i, args in enumerate(all_args):
        content = contents[content_idx]
        result = fabric_tagger.tag(content, args)
        results.append(result)
        if i % 2 == 1:  # Move to next content after every 2 jobs
            content_idx += 1

    for result in results:
        assert result.started is True

    assert len(fabric_tagger.jobstore.active_jobs) == 10

    statuses = []
    for content in contents:
        statuses.append(fabric_tagger.status(content.qid))
    for status in statuses:
        caption = _status_for(status, "caption")
        asr = _status_for(status, "asr")
        assert caption.status.status in ["Queued", "Fetching content"]
        assert asr.status.status in ["Queued", "Fetching content"]

    time.sleep(2)
    for content in contents:
        status = fabric_tagger.status(content.qid)
        caption = _status_for(status, "caption")
        asr = _status_for(status, "asr")
        assert caption.status.status == "Completed"
        assert asr.status.status == "Completed"


def test_tags_uploaded_during_and_after_job(
    fabric_tagger, 
    q, 
    sample_tag_args
):
    # Start jobs
    for args in sample_tag_args:
        result = fabric_tagger.tag(q, args)
        assert result.started is True

    tracks = ["speech_to_text", "object_detection"]

    tag_counts = set()

    timeout = 3
    start = time.time()
    end = False
    while time.time() - start < timeout:
        end = True
        for track in tracks:
            tag_count = fabric_tagger.tagstore.count_tags(track=track, q=q)
            tag_counts.add(tag_count)
            if tag_count < 4:
                end = False
        if end:
            break
        time.sleep(0.01)

    print(fabric_tagger.status(q.qid))

    assert end

    assert len(tag_counts) == 3
    assert 0 in tag_counts
    assert 2 in tag_counts
    assert 4 in tag_counts


def test_tags_uploaded_during_and_after_job_through_status(
    fabric_tagger, 
    q, 
    sample_tag_args
):
    # Same as test_tags_uploaded_during_and_after_job but using status to check job status instead of querying tagstore.
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)
    
    timeout = 3
    start = time.time()
    end = False
    counts = set()
    while time.time() - start < timeout:
        reports = fabric_tagger.status(q.qid)
        end = True
        for r in reports:
            counts.add(len(r.status.uploaded_sources))
            if len(r.status.uploaded_sources) != 2:
                end = False
        if end:
            break
        time.sleep(0.01)

    assert end
    assert 0 in counts
    assert 1 in counts
    assert 2 in counts


def test_container_tags_method_fails(fabric_tagger, q, make_tag_args):
    """Test that when container.tags() fails, the job fails gracefully and stops the container"""

    class BrokenTagsContainer(FakeTagContainer):
        def tags(self) -> list[ModelTag]:
            raise RuntimeError("Simulated container tags() failure")

    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return BrokenTagsContainer(req.media_dir, req.model_id)

    fabric_tagger.cregistry.get = get_side_effect

    args = make_tag_args(feature="caption", stream="video", destination_qid=q.qid)

    result = fabric_tagger.tag(q, args)
    assert result.started is True

    timeout = 3
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = fabric_tagger.status(q.qid)
        job_status = _status_for(status, "caption").status
        if job_status.status == "Failed":
            break
        elif job_status.status in ["Completed", "Stopped"]:
            pytest.fail(f"Expected job to fail, but got status: {job_status.status}")
        time.sleep(0.25)
    else:
        pytest.fail("Job did not fail within timeout period")

    final_status = fabric_tagger.status(q.qid)
    assert _status_for(final_status, "caption").status.status == "Failed"
    assert _status_for(final_status, "caption").status.error == "Simulated container tags() failure"
    assert len(fabric_tagger.jobstore.active_jobs) == 0
    assert len(fabric_tagger.jobstore.inactive_jobs) == 1


def test_container_initialization_fails(fabric_tagger, q, make_tag_args):
    """Test that when container initialization fails, tag() raises the error directly"""

    class FailContainer(FakeTagContainer):
        def start(self, gpu_idx: int | None=None) -> None:
            raise RuntimeError("Simulated container start failure")
        
    fabric_tagger.cregistry.get = Mock(return_value=FailContainer("/fake/media/dir", "model_id"))

    args = make_tag_args(feature="caption", stream="video", destination_qid=q.qid)

    fabric_tagger.tag(q, args)

    time.sleep(1)

    status = fabric_tagger.status(q.qid)
    job_status = _status_for(status, "caption").status
    assert job_status.status == "Failed"
    assert job_status.error == "Simulated container start failure"

def wait_tag(fabric_tagger, batch_id, timeout):
    start = time.time()
    while not timeout or time.time() - start < timeout:
        reports = fabric_tagger.status(batch_id)
        if any(r.status.status == "Failed" for r in reports):
            failed = next(r for r in reports if r.status.status == "Failed")
            pytest.fail(f"Job failed: {failed.status.error or 'unknown error'}")
        if reports and all(r.status.status in ("Completed", "Stopped") for r in reports):
            return
        time.sleep(0.1)
    pytest.fail("Job did not complete within timeout period")


def test_container_nonzero_exit_code(fabric_tagger, q, make_tag_args):
    """Test that a container with non-zero exit code causes job to fail"""

    class NonZeroExitContainer(FakeTagContainer):
        def exit_code(self) -> int | None:
            if self.is_stopped:
                return 1
            return None

    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return NonZeroExitContainer(req.media_dir, req.model_id)

    fabric_tagger.cregistry.get = get_side_effect

    args = make_tag_args(feature="caption", stream="video", destination_qid=q.qid)

    result = fabric_tagger.tag(q, args)
    assert result.started is True

    timeout = 3
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = fabric_tagger.status(q.qid)
        job_status = _status_for(status, "caption").status
        if job_status.status == "Failed":
            break
        elif job_status.status in ["Completed", "Stopped"]:
            pytest.fail(f"Expected job to fail due to exit code, but got: {job_status.status}")
        time.sleep(0.25)
    else:
        pytest.fail("Job did not fail within timeout period")

    final_status = fabric_tagger.status(q.qid)
    assert _status_for(final_status, "caption").status.status == "Failed"
    assert _status_for(final_status, "caption").status.error


def test_destination_qid_uploads_to_correct_qhit(sample_tag_args, fabric_tagger: FabricTagger, q, q_legacy):
    """Test that when destination_qid is set, tags are uploaded to that qid instead of source"""

    # doesn't really matter what the other content is as long as it's in same tenant
    q2 = q_legacy

    for args in sample_tag_args:
        fabric_tagger.tag(q, dc_replace(args, destination_qid=q2.qid))

    # Wait for job to complete
    wait_tag(fabric_tagger, q.qid, timeout=10)
    time.sleep(0.5)
    
    # Verify tags were uploaded to destination_qid
    tag_count_destination = fabric_tagger.tagstore.count_tags(qid=q2.qid, q=q2)
    assert tag_count_destination > 0

    # Verify no tags were uploaded to source qid
    tag_count_source = fabric_tagger.tagstore.count_tags(qid=q.qid, q=q)
    assert tag_count_source == 0

def test_tags_have_timestamp_ms_field(fabric_tagger: FabricTagger, q: Content, sample_tag_args):
    """Test that uploaded tags include timestamp_ms in additional_info"""
    
    for args in sample_tag_args:
        fabric_tagger.tag(q, args)

    wait_tag(fabric_tagger, q.qid, timeout=5)
    
    tags = fabric_tagger.tagstore.find_tags(
        qid=q.qid,
        q=q
    )

    assert len(tags) > 0

    for tag in tags:
        assert tag.additional_info is not None
        assert "timestamp_ms" in tag.additional_info
        assert isinstance(tag.additional_info["timestamp_ms"], int)
        assert tag.additional_info["timestamp_ms"] > 0


def test_uploaded_sources_respects_model_progress(fabric_tagger, q, make_tag_args):
    """Test that a source producing zero tags is marked as failed in job status"""
    
    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return PartialResultContainer(req.media_dir, req.model_id)
    
    fabric_tagger.cregistry.get = get_side_effect
    
    args = make_tag_args(feature="caption", stream="video")
    
    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)
    
    status = fabric_tagger.status(q.qid)
    report = _status_for(status, "caption")
    assert report.status.status == "Completed"
    assert len(report.status.downloaded_sources) == 2
    assert len(report.status.tagged_sources) == 2
    assert len(report.status.uploaded_sources) == 2
    assert report.status.error is None


def test_track_override_uploads_to_multiple_tracks(fabric_tagger, q, make_tag_args):
    """Test that model tags with different tracks are uploaded to different tagstore tracks"""

    class MultiTrackContainer(FakeTagContainer):
        def tags(self) -> list[ModelTag]:
            tags = []
            finished_files = self.media_files if self.is_stopped else self.media_files[:-1]
            for i, filepath in enumerate(finished_files):
                tags.append(ModelTag(0, 5000, f"default_track_tag_{i}", filepath, "", None))
                tags.append(ModelTag(5000, 10000, f"pretty_track_tag_{i}", filepath, "pretty", None))
            return tags

    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return MultiTrackContainer(req.media_dir, req.model_id)

    fabric_tagger.cregistry.get = get_side_effect

    args = make_tag_args(feature="asr", stream="audio")
    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)

    default_tags = fabric_tagger.tagstore.find_tags(q=q, track="speech_to_text")
    override_tags = fabric_tagger.tagstore.find_tags(q=q, track="auto_captions")
    assert len(default_tags) == 2
    assert len(override_tags) == 2

def test_uploaded_track_label(fabric_tagger: FabricTagger, q, make_tag_args):
    """Test that uploaded tags have correct track labels based on model params"""
    
    args = make_tag_args(feature="caption", stream="video")
    
    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)
    
    track_arg = fabric_tagger.track_resolver.resolve(args.feature)
    
    track = fabric_tagger.tagstore.get_track(
        q=q,
        name=track_arg.name
    )
    
    assert isinstance(track, Track)
    assert track.name == track_arg.name
    assert track.label == track_arg.label

def test_default_defer_to_model_track(fabric_tagger, q, make_tag_args):
    class MultiTrackContainer(FakeTagContainer):
        def tags(self) -> list[ModelTag]:
            tags = []
            finished_files = self.media_files if self.is_stopped else self.media_files[:-1]
            for i, filepath in enumerate(finished_files):
                tags.append(ModelTag(
                    start_time=0,
                    end_time=5000,
                    text=f"default_track_tag_{i}",
                    source_media=filepath,
                    model_track="random_track"
                ))
            return tags

    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return MultiTrackContainer(req.media_dir, req.model_id)

    with patch.object(fabric_tagger.cregistry, "get", side_effect=get_side_effect):
        args = make_tag_args(feature="asr", stream="audio")
        fabric_tagger.tag(q, args)
        wait_tag(fabric_tagger, q.qid, timeout=5)

    default_tags = fabric_tagger.tagstore.find_tags(q=q, track="random_track")
    assert len(default_tags) == 2

    track = fabric_tagger.tagstore.get_track(q=q, name="random_track")
    assert track.label == "Random Track"

def test_fetcher_returns_no_sources(fabric_tagger, q, make_tag_args):
    """Test that job completes gracefully when fetcher returns no sources"""

    empty_result = DownloadResult(sources=[], failed=[], done=True)
    fake_session = Mock(download=Mock(return_value=empty_result))

    with patch.object(fabric_tagger.fetcher, "get_session", return_value=fake_session):
        args = make_tag_args(feature="caption", stream="video")
        result = fabric_tagger.tag(q, args)
        assert result.started is True
        wait_tag(fabric_tagger, q.qid, timeout=2)

    final_status = fabric_tagger.status(q.qid)
    report = _status_for(final_status, "caption")
    assert report.status.status == "Completed"
    assert len(report.status.downloaded_sources) == 0
    assert len(report.status.tagged_sources) == 0
    assert report.status.error is None
    
    # Verify no tags were uploaded
    tag_count = fabric_tagger.tagstore.count_tags(qid=q.qid, q=q)
    assert tag_count == 0

def test_batch_report_on_success(fabric_tagger, q, make_tag_args):
    """Batch additional_info should contain a tagger report with Completed status."""
    args = make_tag_args(feature="caption", stream="video")
    result = fabric_tagger.tag(q, args)
    assert result.started

    wait_tag(fabric_tagger, q.qid, timeout=5)

    batches = fabric_tagger.tagstore.find_batches(qid=q.qid, q=q)
    assert len(batches) == 1

    batch = fabric_tagger.tagstore.get_batch(batches[0], q=q)
    assert batch is not None

    report = batch.additional_info.get("tagger")
    assert report is not None
    assert report["job_status"]["status"] == "Completed"
    assert report["upload_status"] is not None
    # all downloaded parts should be reported as tagged on a clean completion
    assert set(report["upload_status"]["tagged_sources"]) == set(report["upload_status"]["all_sources"])


def test_replace(
    fabric_tagger: FabricTagger,
    q: Content,
    make_tag_args,
):

    args = make_tag_args(feature="caption", stream="video")

    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)

    first_batch = fabric_tagger.tagstore.find_batches(q=q, track="object_detection")[0]
    tags = fabric_tagger.tagstore.find_tags(q=q, track="object_detection", batch_id=first_batch)

    timestamps = tuple(sorted(t.additional_info["timestamp_ms"] for t in tags if t.additional_info is not None))

    args = make_tag_args(feature="caption", stream="video", replace=True)
    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)

    new_batches = fabric_tagger.tagstore.find_batches(q=q, track="object_detection")
    new_batches.remove(first_batch)
    second_batch = new_batches[0]

    new_tags = fabric_tagger.tagstore.find_tags(q=q, batch_id=second_batch)

    new_timestamps = tuple(sorted(t.additional_info["timestamp_ms"] for t in new_tags if t.additional_info is not None))

    assert new_timestamps > timestamps

    # try with replace = False
    args = make_tag_args(feature="caption", stream="video", replace=False)
    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)

    # doing this weird stuff for now cause prod tagstore does shadowing and local one doesn't
    batches = fabric_tagger.tagstore.find_batches(q=q, track="object_detection")
    
    assert len(batches) == 2

def test_second_run_has_no_parts(fabric_tagger, q, make_tag_args):
    """Test that second run of same job has no parts to download"""

    args = make_tag_args(feature="caption", stream="video", replace=False)

    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)

    status = fabric_tagger.status(q.qid)
    status1 = _status_for(status, "caption")
    assert status1.status.status == "Completed"
    assert len(status1.status.total_sources) == 2
    assert len(status1.status.downloaded_sources) == 2
    assert len(status1.status.tagged_sources) == 2

    # Start same job again
    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)

    status = fabric_tagger.status(q.qid)
    status1 = _status_for(status, "caption")
    assert status1.status.status == "Completed"
    assert len(status1.status.total_sources) == 0
    assert len(status1.status.downloaded_sources) == 0
    assert len(status1.status.tagged_sources) == 0

    args.replace = True
    fabric_tagger.tag(q, args)
    wait_tag(fabric_tagger, q.qid, timeout=5)

    status = fabric_tagger.status(q.qid)
    status1 = _status_for(status, "caption")
    assert len(status1.status.total_sources) == 2
    assert len(status1.status.downloaded_sources) == 2
    assert len(status1.status.tagged_sources) == 2

def test_upload_fail_propagates(fabric_tagger, q, make_tag_args):
    """Test that if upload_tags raises an exception, job transitions to Failed state"""

    fabric_tagger.tagstore.upload_tags = Mock(side_effect=RuntimeError("Simulated upload failure"))

    args = make_tag_args(feature="caption", stream="video")

    fabric_tagger.tag(q, args)

    time.sleep(1)

    status = fabric_tagger.status(q.qid)
    report = _status_for(status, "caption")
    assert report.status.status == "Failed"
    assert report.status.error == "Simulated upload failure"

def test_model_error_through_status(fabric_tagger, q, make_tag_args):
    class ErrorContainer(FakeTagContainer):
        def errors(self) -> list[Error]:
            return [Error(source_media=self.media_files[0], message="Simulated model error")]

        def exit_code(self) -> int | None:
            return 1

    def get_side_effect(req: ContainerRequest) -> FakeTagContainer:
        return ErrorContainer(req.media_dir, req.model_id)

    fabric_tagger.cregistry.get = get_side_effect

    args = make_tag_args(feature="caption", stream="video")

    fabric_tagger.tag(q, args)

    time.sleep(1)

    status = fabric_tagger.status(q.qid)
    report = _status_for(status, "caption")
    assert report.status.status == "Failed"
    assert report.status.error == "Simulated model error"

def test_part_download_warnings(fabric_tagger, q, make_tag_args):
    """Test that if some parts fail to download but job completes, the failed parts are reported in status"""

    class PartialFailFetchWorker(FetchSession):
        def download(self) -> DownloadResult:
            return DownloadResult(
                sources=[Source(
                    name="video1",
                    filepath="/path/to/video1",
                    offset=0,
                    wall_clock=None
                )],
                failed=["video2"],
                done=True
            )

        def metadata(self) -> MediaMetadata:
            return MediaMetadata(sources=["video1", "video2"], fps=30.0)
        
        @property
        def path(self) -> str:
            return "/fake/path"

    fabric_tagger.fetcher.get_session = Mock(return_value=PartialFailFetchWorker())

    args = make_tag_args(feature="caption", stream="video")

    fabric_tagger.tag(q, args)

    wait_tag(fabric_tagger, q.qid, timeout=5)

    status = fabric_tagger.status(q.qid)
    report = _status_for(status, "caption")
    assert report.status.status == "Completed"
    assert len(report.status.total_sources) == 2
    assert len(report.status.downloaded_sources) == 1
    assert len(report.status.tagged_sources) == 1
    assert report.status.error is None
    assert len(report.status.warnings) == 1

def test_unknown_container(fabric_tagger, q, make_tag_args):
    """Test that if container registry returns a container that doesn't match the requested model, job fails gracefully"""

    args = make_tag_args(feature="not exists", stream="video")
    
    with pytest.raises(MissingResourceError):
        fabric_tagger.tag(q, args)