import json
import os
import subprocess
from functools import cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@cache
def build_info() -> dict:
    """Revision from the local git checkout, else from version/buildinfo.json (written by the image build, which has no .git)."""
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return {"revision": revision}
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with open(os.path.join(ROOT, "version", "buildinfo.json")) as f:
            info = json.load(f)
        return {k: info[k] for k in ("revision", "branch", "build_date") if info.get(k)}
    except (OSError, ValueError):
        return {}
