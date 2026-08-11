"""Uploader dispatch: which of the tagstore / vectorstore a model's outputs land in.

Exercised against the filesystem tagstore and the mock vectorstore.
"""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from src.common.errors import BadRequestError
from src.tagging.fabric_tagging.model import UploadStatus
from src.tagging.uploading.uploader import Uploader
from src.tags.datastore.model import Tag


@dataclass
class MockReport:
    """Stand-in for TagContentStatusReport, only upload_status is read back."""
    upload_status: UploadStatus


def _report(sources: list[str]) -> MockReport:
    return MockReport(upload_status=UploadStatus(
        all_sources=sources,
        downloaded_sources=sources,
        tagged_sources=sources,
        uploaded_sources=sources,
    ))


def test_text_tags_go_to_the_tagstore(uploader, vector_store, get_tag):
    ts = uploader.tag_session.datastore
    q = uploader.dest_q

    uploader.upload(
        tags=[get_tag(model_track="", data="hello world", source_media="source1")],
        tagged_sources=["source1"],
    )

    assert {t.data for t in ts.find_tags(q=q, track="speech_to_text")} == {"hello world"}
    assert vector_store.find_tags(q=q) == []


def test_vectors_go_to_the_vectorstore(uploader, vector_store, get_vector_tag):
    ts = uploader.tag_session.datastore
    q = uploader.dest_q

    uploader.upload(
        tags=[get_vector_tag(model_track="embedding", data=[1.0, 2.0], source_media="source1")],
        tagged_sources=["source1"],
    )

    vectors = vector_store.find_tags(q=q, track="embedding")
    assert [v.data for v in vectors] == [[1.0, 2.0]]
    assert ts.find_tags(q=q) == []


def test_mixed_output_is_split_across_both_stores(uploader, vector_store, get_tag, get_vector_tag):
    ts = uploader.tag_session.datastore
    q = uploader.dest_q

    uploader.upload(
        tags=[
            get_tag(model_track="speech_to_text", data="hello", source_media="source1"),
            get_vector_tag(model_track="embedding", data=[0.5], source_media="source1"),
        ],
        tagged_sources=["source1"],
    )

    assert {t.data for t in ts.find_tags(q=q, track="speech_to_text")} == {"hello"}
    assert [v.data for v in vector_store.find_tags(q=q, track="embedding")] == [[0.5]]


def test_vectors_without_a_vectorstore_raise(tagstore_only_uploader, get_vector_tag):
    """A job started without an index_qid cannot store the model's embeddings."""
    with pytest.raises(BadRequestError, match="index_qid"):
        tagstore_only_uploader.upload(
            tags=[get_vector_tag(model_track="embedding", source_media="source1")],
            tagged_sources=["source1"],
        )


def test_report_goes_to_the_tagstore_for_a_text_model(uploader, vector_store, get_tag):
    ts = uploader.tag_session.datastore
    q = uploader.dest_q

    uploader.upload(
        tags=[get_tag(model_track="", data="hello", source_media="source1")],
        tagged_sources=["source1"],
    )
    uploader.upload_report(_report(["source1"]))  # type: ignore

    tag_batches = ts.find_batches(q=q, model="asr")
    assert len(tag_batches) == 1
    assert tag_batches[0].additional_info["tagger"]["upload_status"]["uploaded_sources"] == ["source1"]
    assert vector_store.find_batches(q=q, model="asr") == []


def test_report_goes_to_the_vectorstore_for_a_vector_model(uploader, vector_store, get_vector_tag):
    ts = uploader.tag_session.datastore
    q = uploader.dest_q

    uploader.upload(
        tags=[get_vector_tag(model_track="embedding", source_media="source1")],
        tagged_sources=["source1"],
    )
    uploader.upload_report(_report(["source1"]))  # type: ignore

    vector_batches = vector_store.find_batches(q=q, model="asr")
    assert len(vector_batches) == 1
    assert vector_batches[0].additional_info["tagger"]["upload_status"]["uploaded_sources"] == ["source1"]
    # no text tags were produced, so the tagstore holds nothing for this run
    assert ts.find_batches(q=q, model="asr") == []


def test_report_goes_to_both_stores_for_a_mixed_model(uploader, vector_store, get_tag, get_vector_tag):
    ts = uploader.tag_session.datastore
    q = uploader.dest_q

    uploader.upload(
        tags=[
            get_tag(model_track="speech_to_text", data="hello", source_media="source1"),
            get_vector_tag(model_track="embedding", source_media="source1"),
        ],
        tagged_sources=["source1"],
    )
    uploader.upload_report(_report(["source1"]))  # type: ignore

    assert len(ts.find_batches(q=q, model="asr")) == 1
    assert len(vector_store.find_batches(q=q, model="asr")) == 1


def test_report_defaults_to_the_tagstore_when_nothing_was_produced(uploader, vector_store):
    """A run that tags a source but emits nothing still leaves a batch behind."""
    ts = uploader.tag_session.datastore
    q = uploader.dest_q

    uploader.upload(tags=[], tagged_sources=["source1"])
    uploader.upload_report(_report(["source1"]))  # type: ignore

    assert len(ts.find_batches(q=q, model="asr")) == 1
    assert vector_store.find_batches(q=q, model="asr") == []


def test_reupload_replaces_vectors_for_a_source(uploader, vector_store, get_vector_tag, track_resolver, mock_q, filesystem_tagstore):
    q = uploader.dest_q

    uploader.upload(
        tags=[
            get_vector_tag(model_track="embedding", data=[1.0], source_media="source1"),
            get_vector_tag(model_track="embedding", data=[2.0], source_media="source2"),
        ],
        tagged_sources=["source1", "source2"],
    )
    assert {tuple(v.data) for v in vector_store.find_tags(q=q)} == {(1.0,), (2.0,)}

    # a fresh run reprocesses only source1
    second = Uploader(
        feature="asr",
        track_resolver=track_resolver,
        tagstore=filesystem_tagstore,
        vectorstore=vector_store,
        dest_q=mock_q,
        track_suffix="",
        do_retry=False,
    )
    second.upload(
        tags=[get_vector_tag(model_track="embedding", data=[3.0], source_media="source1")],
        tagged_sources=["source1"],
    )

    assert {tuple(v.data) for v in vector_store.find_tags(q=q, sources=["source1"])} == {(3.0,)}
    assert {tuple(v.data) for v in vector_store.find_tags(q=q, sources=["source2"])} == {(2.0,)}


def test_uploaded_sources_requires_every_store_to_succeed(uploader, vector_store, get_tag, get_vector_tag):
    """A vectorstore failure must not let a source be reported as uploaded, otherwise
    a diff-based re-run would skip it."""
    vector_store.upload_tags = Mock(side_effect=Exception("vectorstore down"))

    with pytest.raises(Exception, match="vectorstore down"):
        uploader.upload(
            tags=[
                get_tag(model_track="speech_to_text", data="hello", source_media="source1"),
                get_vector_tag(model_track="embedding", source_media="source1"),
            ],
            tagged_sources=["source1"],
        )

    assert uploader.get_uploaded_sources() == []


def test_uploaded_sources_unions_the_progress_of_both_stores(uploader, get_tag, get_vector_tag):
    uploader.upload(
        tags=[
            get_tag(model_track="speech_to_text", data="hello", source_media="source1"),
            get_vector_tag(model_track="embedding", source_media="source2"),
        ],
        tagged_sources=["source1", "source2"],
    )

    assert set(uploader.get_uploaded_sources()) == {"source1", "source2"}


def test_tagstore_rejects_vectors(filesystem_tagstore, q):
    """Guard against a vector slipping past the dispatch and corrupting a tagstore."""
    batch = filesystem_tagstore.create_batch(model="asr", author="tagger", q=q)
    with pytest.raises(ValueError):
        filesystem_tagstore.upload_tags(
            [Tag(id="", start_time=0, end_time=1, data=[0.1], additional_info=None, source="s", batch_id=batch.id)],
            batch.id,
            track="embedding",
            q=q,
        )


def test_vectorstore_rejects_text(vector_store, q):
    batch = vector_store.create_batch(model="asr", author="tagger", q=q)
    with pytest.raises(ValueError):
        vector_store.upload_tags(
            [Tag(id="", start_time=0, end_time=1, data="hello", additional_info=None, source="s", batch_id=batch.id)],
            batch.id,
            track="embedding",
            q=q,
        )
