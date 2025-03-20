from typing import Dict

def parse_qhit(qhit: str) -> Dict[str, str]:
    """Parse a qhit into a dictionary so it can be passed to elv_client_py functions and use the correct argument."""
    if qhit.startswith("iq__"):
        return {"object_id": qhit}
    elif qhit.startswith("hq__"):
        return {"version_hash": qhit}
    elif qhit.startswith("tqw__"):
        return {"write_token": qhit}
    raise ValueError(f"Invalid qhit: {qhit}")