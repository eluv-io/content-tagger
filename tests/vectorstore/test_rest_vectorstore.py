"""RestVectorstore against a live elv-vectorstore.

Configure with:
    TEST_VECTORSTORE_HOST  base url of the service (required, tests skip without it)
    TEST_AUTH              token, shared with the tagstore tests (required)

The index is created on the configured content if it does not exist yet, and every
batch a test writes is deleted on teardown, so a run leaves the index as it found it.

Writes are verified through the service's own read endpoints (`/search`, `/tracks`),
since the vectorstore never lists what it holds: a batch can be written, amended and
deleted but not read back, and vectors are only reachable by search.
"""

import pytest

from src.tags.datastore.model import Batch, Tag, Track
from tests.vectorstore.conftest import SOURCES


def _upload(store, q, model, track, tags) -> str:
    """Write tags under a fresh batch, the way an UploadSession does."""
    batch = store.create_batch(model=model, author="tagger", q=q)
    store.upload_tags(tags, batch.id, track, q=q)
    return batch.id


def test_create_batch(live_vectorstore, q, vector_model):
    batch = live_vectorstore.create_batch(model=vector_model, author="tagger", q=q)

    assert isinstance(batch, Batch)
    assert batch.id
    assert batch.qid == q.qid
    assert batch.model == vector_model
    assert batch.author == "tagger"


def test_uploaded_vectors_are_searchable(live_vectorstore, vector_api, q, vector_model, make_vector):
    track = f"{vector_model}_embedding"
    tag = make_vector(seed=1.0, source=SOURCES[0], start_time=100, end_time=200)

    batch_id = _upload(live_vectorstore, q, vector_model, track, [tag])

    hits = vector_api.search(tag.data, track=track)
    assert len(hits) == 1
    assert hits[0]["source"] == SOURCES[0]
    assert hits[0]["track"] == track
    assert hits[0]["start_time"] == 100
    assert hits[0]["end_time"] == 200
    assert hits[0]["additional_info"] == tag.additional_info
    # the content and the batch are sent once for the whole request, not per vector
    assert hits[0]["qid"] == q.qid
    assert hits[0]["batch_id"] == batch_id


def test_uploaded_vectors_are_counted_on_the_track(live_vectorstore, vector_api, q, vector_model, make_vector):
    track = f"{vector_model}_embedding"
    assert track not in vector_api.tracks()

    _upload(live_vectorstore, q, vector_model, track, [
        make_vector(seed=1.0, source=SOURCES[0]),
        make_vector(seed=2.0, source=SOURCES[1]),
    ])

    assert vector_api.tracks().get(track) == 2


def test_frame_index_round_trips(live_vectorstore, vector_api, q, vector_model, make_vector):
    track = f"{vector_model}_embedding"
    tag = make_vector(seed=1.0, frame_info={"frame_idx": 42, "box": {"x1": 0.1}})

    _upload(live_vectorstore, q, vector_model, track, [tag])

    hits = vector_api.search(tag.data, track=track)
    assert len(hits) == 1
    assert hits[0]["frame_idx"] == 42


def test_short_vectors_are_padded(live_vectorstore, vector_api, q, vector_model, make_vector, vector_size):
    """A model emitting a smaller embedding than the index is configured for still writes:
    the service rejects any vector whose dimension does not match the index exactly."""
    track = f"{vector_model}_embedding"
    tag = make_vector(seed=1.0, dims=vector_size // 2)

    _upload(live_vectorstore, q, vector_model, track, [tag])

    assert vector_api.tracks().get(track) == 1
    # padding happens on the way out, the caller's tag is left as the model produced it
    assert len(tag.data) == vector_size // 2
    hits = vector_api.search(tag.data + [0.0] * (vector_size - len(tag.data)), track=track)
    assert len(hits) == 1
    assert hits[0]["source"] == SOURCES[0]


def test_oversized_vectors_are_refused(live_vectorstore, vector_api, q, vector_model, make_vector, vector_size):
    """Trimming a too-long vector would silently corrupt the embedding, so nothing is sent."""
    track = f"{vector_model}_embedding"
    batch = live_vectorstore.create_batch(model=vector_model, author="tagger", q=q)

    with pytest.raises(ValueError):
        live_vectorstore.upload_tags([make_vector(seed=1.0, dims=vector_size + 1)], batch.id, track, q=q)

    assert track not in vector_api.tracks()


def test_empty_vectors_are_refused(live_vectorstore, vector_api, q, vector_model, make_vector):
    """All-zero padding of an empty vector would store a vector carrying no embedding."""
    track = f"{vector_model}_embedding"
    batch = live_vectorstore.create_batch(model=vector_model, author="tagger", q=q)

    with pytest.raises(ValueError):
        live_vectorstore.upload_tags([make_vector(seed=0.0, dims=0)], batch.id, track, q=q)

    assert track not in vector_api.tracks()


def test_upload_rejects_text(live_vectorstore, q, vector_model):
    text_tag = Tag(
        id="", start_time=0, end_time=1, data="hello",
        additional_info=None, source=SOURCES[0], batch_id="", frame_info=None,
    )

    with pytest.raises(ValueError):
        live_vectorstore.upload_tags([text_tag], "some_batch", f"{vector_model}_embedding", q=q)


def test_upload_without_a_batch_is_refused(live_vectorstore, vector_api, q, vector_model, make_vector):
    """The service would accept it, but a vector with no batch matches no delete and so
    could never be removed from the index."""
    track = f"{vector_model}_embedding"

    with pytest.raises(ValueError):
        live_vectorstore.upload_tags([make_vector(seed=1.0)], "", track, q=q)

    assert track not in vector_api.tracks()


def test_upload_empty_is_a_noop(live_vectorstore, vector_api, q, vector_model):
    track = f"{vector_model}_embedding"

    live_vectorstore.upload_tags([], "some_batch", track, q=q)

    assert track not in vector_api.tracks()


def test_update_batch_is_accepted(live_vectorstore, q, vector_model):
    """Writing the run report onto the batch, which is how a vector model's report
    reaches the index."""
    batch = live_vectorstore.create_batch(model=vector_model, author="tagger", q=q)

    live_vectorstore.update_batch(
        batch_id=batch.id,
        additional_info={"tagger": {"upload_status": {"uploaded_sources": SOURCES}}},
        q=q,
    )


def test_delete_by_source_leaves_other_sources(live_vectorstore, vector_api, q, vector_model, make_vector):
    track = f"{vector_model}_embedding"
    _upload(live_vectorstore, q, vector_model, track, [
        make_vector(seed=1.0, source=SOURCES[0]),
        make_vector(seed=2.0, source=SOURCES[1]),
    ])
    assert vector_api.tracks().get(track) == 2

    live_vectorstore.delete_tags_by_source(sources=[SOURCES[0]], model=vector_model, q=q)

    assert vector_api.tracks().get(track) == 1
    remaining = vector_api.search(make_vector(seed=2.0).data, track=track)
    assert [hit["source"] for hit in remaining] == [SOURCES[1]]


def test_delete_is_scoped_to_the_model(live_vectorstore, vector_api, q, vector_model, make_vector):
    """The same source embedded by two models must not be clobbered by one of them."""
    other_model = f"{vector_model}_other"
    track = f"{vector_model}_embedding"
    other_track = f"{vector_model}_other_embedding"

    _upload(live_vectorstore, q, vector_model, track, [make_vector(seed=1.0, source=SOURCES[0])])
    _upload(live_vectorstore, q, other_model, other_track, [make_vector(seed=2.0, source=SOURCES[0])])

    live_vectorstore.delete_tags_by_source(sources=[SOURCES[0]], model=vector_model, q=q)

    tracks = vector_api.tracks()
    assert track not in tracks
    assert tracks.get(other_track) == 1


@pytest.mark.parametrize("sources,model", [([], "some_model"), (["s1"], "")])
def test_delete_without_both_filters_is_a_noop(live_vectorstore, vector_api, q, vector_model, make_vector, sources, model):
    """The endpoint never matches vectors lacking a batch or a source, so an unscoped
    delete is refused client-side rather than issued and silently doing nothing."""
    track = f"{vector_model}_embedding"
    _upload(live_vectorstore, q, vector_model, track, [make_vector(seed=1.0, source=SOURCES[0])])

    live_vectorstore.delete_tags_by_source(sources=sources, model=model, q=q)

    assert vector_api.tracks().get(track) == 1


def test_delete_batch_removes_its_vectors(live_vectorstore, vector_api, q, vector_model, make_vector):
    """How a run is cleared wholesale, and how these tests clean up after themselves."""
    track = f"{vector_model}_embedding"
    batch_id = _upload(live_vectorstore, q, vector_model, track, [
        make_vector(seed=1.0, source=SOURCES[0]),
        make_vector(seed=2.0, source=SOURCES[1]),
    ])
    assert vector_api.tracks().get(track) == 2

    live_vectorstore.delete_batch(batch_id, q=q)

    assert track not in vector_api.tracks()


def test_tracks_are_stubbed(live_vectorstore, vector_api, q, vector_model):
    """A vectorstore has no first class track, each vector carries its own name, so
    create_track writes nothing and get_track always reports the track as present."""
    name = f"{vector_model}_never_written"

    live_vectorstore.create_track(name=name, label="Never Written", q=q)
    track = live_vectorstore.get_track(name=name, q=q)

    assert isinstance(track, Track)
    assert track.name == name
    assert track.qid == q.qid
    assert name not in vector_api.tracks()


@pytest.mark.parametrize("call", [
    lambda s, q: s.find_tags(q, track="embedding"),
    lambda s, q: s.count_tags(q, track="embedding"),
    lambda s, q: s.get_batch("00000000-0000-0000-0000-000000000000", q=q),
    lambda s, q: s.find_batches(q, model="a_model"),
])
def test_unsupported_operations_raise(live_vectorstore, q, call):
    """The read half of the Datastore protocol: the service lists neither vectors nor
    batches, so these fail loudly rather than reporting an index as empty."""
    with pytest.raises(NotImplementedError):
        call(live_vectorstore, q)
