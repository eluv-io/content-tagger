from server import reload_model_configs
from src.common.model import ModelConfigStore
from src.tag_containers.model import ModelConfig
from src.tags.track_resolver import LabelResolverConfig, TrackResolver


def _cfg(image: str, track_outputs: list[str]) -> ModelConfig:
    return ModelConfig(image=image, description="desc", type="frame", resources={"gpu": 1}, track_outputs=track_outputs)


def test_reload_swaps_configs_and_tracks():
    old_kept = _cfg("img:1", ["kept_track"])
    store = ModelConfigStore({"kept": old_kept, "gone": _cfg("img:1", ["gone_track"])})
    track_resolver = TrackResolver(LabelResolverConfig(mapping={}), store)
    before = store.snapshot()

    reload_model_configs(store, {"kept": _cfg("img:2", ["kept_track"]), "new": _cfg("img:1", ["new_track"])})

    after = store.snapshot()
    assert set(after) == {"kept", "new"}
    assert after["kept"].image == "img:2"
    # snapshots taken before the reload are unchanged
    assert set(before) == {"kept", "gone"}
    assert old_kept.image == "img:1"

    assert [t.name for t in track_resolver.resolve("new")] == ["new_track"]
    assert track_resolver.reverse_resolve("new_track") == ["new"]
    # removed model falls back to the default track
    assert [t.name for t in track_resolver.resolve("gone")] == ["gone"]
    assert track_resolver.reverse_resolve("gone_track") == ["gone_track"]


def test_store_copies_input_dict():
    configs = {"a": _cfg("img:1", [])}
    store = ModelConfigStore(configs)
    configs["b"] = _cfg("img:1", [])
    assert set(store.snapshot()) == {"a"}
