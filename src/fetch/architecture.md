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
    - implements a `download` method which downloads a single batch and a flag indicating whether more is coming or not.
- `FetchFactory`
    - Factory for the `DownloadWorker`s
    - Currently it depends on the `Tagstore` so that it can query for already tagged sources, might deprecate this cause arguable the factory is doing too much. 
    - Using a factory is nice cause instantiating the workers has semi-complex logic. Must query metadata, select the worker class based on the request context (asset, live, or vod). Must query tagstore to figure out which media to filter. It also makes mocking and DI easier. 

See `model.py`

