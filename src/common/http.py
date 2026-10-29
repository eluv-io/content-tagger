import re

import requests

MAX_BODY_CHARS = 2000

_AUTH_PARAM = re.compile(r"(authorization=)[^&]*", re.IGNORECASE)

def raise_for_status(response: requests.Response) -> None:
    """Like Response.raise_for_status, but the error message includes the response body."""
    if response.ok:
        return
    body = response.text
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "...(truncated)"
    method = response.request.method if response.request is not None else ""
    url = _AUTH_PARAM.sub(r"\1<redacted>", response.url or "")
    raise requests.HTTPError(
        f"{response.status_code} {response.reason} for {method} {url}: {body}",
        response=response,
    )
