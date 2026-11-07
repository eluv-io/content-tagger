from src.tag_containers.factory import _discover_verticalize_service

def test_discover_verticalize_service():
    url = _discover_verticalize_service()
    assert url.startswith("https://")
