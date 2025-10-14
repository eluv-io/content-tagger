# Media fetching

This module is responsible for everything regarding downloading media from the content fabric. 

## Important entities

- `Source`: Represents a "piece" of the content object - can be a single asset or a part.
- `Scope`: Used in the request to the fetcher to specify the subset of content that will be downloaded. There can be multiple sources which belong to the same scope.
    - e.g. start/end time, or a list of assets. In the case of livestream it's a segment duration which represents "give me the most recent seg_dur worth of video".
- `DownloadWorker`
    - The class that actually downloads the media to the local filesystem. It will return a flag when there is no more media to download.
    - It's useful for carrying all the state associated with downloading media from a content object when running a tag job.
    - Implementing it this way allows us to support starting tagging before downloading the entire media which is especially critical for live tagging.
- `Fetcher`
    - Basically just a factory for the `DownloadWorker`s
    - Currently it depends on the `Tagstore` so that it can query for already tagged sources, but this is deprecated cause it's a violation of responsibility boundaries.
    - It is useful to use a factory here because it helps us test since we can override via 

See `model.py`

