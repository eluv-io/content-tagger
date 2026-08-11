
from dataclasses import dataclass

import pytest

from src.tagging.fabric_tagging.model import UploadStatus
from src.tagging.fabric_tagging.source_resolver import SourceResolver
from src.tagging.uploading.uploader import UploadSession


@dataclass
class MockReport:
    """Minimal stand-in for TagContentStatusReport: only the nested upload_status
    (which carries uploaded_sources) is read back by the SourceResolver."""
    upload_status: UploadStatus


@pytest.fixture
def source_resolver(tag_store, vectorstores, track_resolver):
    return SourceResolver(
        tagstore=tag_store,
        vectorstores=vectorstores,
        track_resolver=track_resolver
    )

def test_source_resolver(q, source_resolver):
    tagstore = source_resolver.tagstore
    batch = tagstore.create_batch(
        q=q,
        author="tagger",
        model="asr",
    )

    tagstore.update_batch(
        batch_id=batch.id,
        additional_info={
            "tagger": {
                "upload_status": {
                    "uploaded_sources": ["source1", "source2"]
                }
            }
        },
        q=q
    )

    assert source_resolver.resolve(q, model="asr") == ["source1", "source2"]


def _run_and_report(tag_store, track_resolver, q, track_suffix, sources, get_tag):
    """Simulate a tagging run: upload one tag per source, then record the report
    (with uploaded_sources) onto the session's batch, exactly as the tagger does."""
    session = UploadSession(
        feature="asr",
        track_resolver=track_resolver,
        datastore=tag_store,
        dest_q=q,
        track_suffix=track_suffix,
        do_retry=False,
    )
    session.upload_tags(
        tags=[get_tag(model_track="", data="t", source_media=s) for s in sources],
        tagged_sources=sources,
    )
    session.upload_report(MockReport( # type: ignore
        upload_status=UploadStatus(
            all_sources=sources,
            downloaded_sources=sources,
            tagged_sources=sources,
            uploaded_sources=session.get_uploaded_sources(),
        )
    ))


def test_source_resolver_respects_track_suffix(q, tag_store, vectorstores, track_resolver, get_tag):
    """A track_suffix scopes the run to a distinct batch model, so a suffixed run's
    sources resolve independently from the unsuffixed run's. This also pins the
    uploader and resolver to the same model-name convention end to end."""
    resolver = SourceResolver(tagstore=tag_store, vectorstores=vectorstores, track_resolver=track_resolver)

    # unsuffixed run tags source1; a "v2" run tags source2
    _run_and_report(tag_store, track_resolver, q, track_suffix="", sources=["source1"], get_tag=get_tag)
    _run_and_report(tag_store, track_resolver, q, track_suffix="v2", sources=["source2"], get_tag=get_tag)

    # each suffix sees only its own sources, so the scopes are disjoint
    assert resolver.resolve(q, model="asr", track_suffix="") == ["source1"]
    assert resolver.resolve(q, model="asr", track_suffix="v2") == ["source2"]
