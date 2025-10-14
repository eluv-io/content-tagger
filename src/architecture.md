# Tagger codebase

## Important components/layer

- API layer (`src/api`)
    - just defines the handlers which call the service components
    - authenticates against the fabric by just doing a qinfo call, thereafter it still needs to download parts/media using the same auth token. In theory these media files could be cached, so really the only barrier to being able to trigger tagging is to call qinfo. But you can't see anything until you upload (requires write token) so it's ok.
- Fetch layer (`src/fetch`)
    - fetches media from the fabric
    - uses a session factory approach. `src/fetch/fetch_content.py` implements `Fetcher` which returns 