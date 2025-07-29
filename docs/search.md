# Search Plan

With the changes suggested, we will be able to create a new index object without specifying anything other than a content type. And we can update it with a simple flow:

1. create a write token against the index object
2. call the update lro
3. finalize

All the changes suggested are backwards compatible with the current design.

Details below...

## Extend the existing "root" logic

The goal is to remove the tedious extra step of having to update a site object, as well as to remove some boilerplate when creating the index config. 

- Currently, we specify a root object to crawl (by qid) and then the crawl lro will resolve the latest version and recursively crawl all the metadata in the root object, resolving any links that are found along the way
- Rather than pointing to one root object which contains all the links (which are versioned), we can just specify a list of root contents. Each "root" is a content we want in our index.
- If only a qlib is specified we will crawl the whole library
- If no root is specified, we crawl the library that the index is contained in.

## Storing index defaults in the content type metadata

At crawl/search time, we first retrieve the config information from the content-type and then we apply any overrides which exist in the index metadata. This way we don't need to specify any metadata in the index. 

This gives us tenant level configurations over different classes of indexes. It is probably a better choice than to auto populate index objects in each tenancy, because it gives us the freedom to make new indexes as needed  just using the frowser, without needing to copy an existing index or write a script to do it. It also gives us a human readable "class" name for the different categories of index. 

Tenant setup would involve setting a few index content-types. Right now we just specify one `Index` type. 

- Example types would be `Index - Clips`, `Index - Assets`, or `Index - Contents`
- Metadata includes the usual doc prefix (aggregation by shot, asset, content-level respectively) as well as the fields and their crawl paths. 
- The metadata in the index object itself overrides whatever is in the type defaults.
- In theory, this way we can build an index by just creating an index with some type in some library and then crawling it
- Setting up a new tenant would involve just making these three canonical index types, which are more or less the same for different Tenants, and then we can create a new index by just creating an index type object in a library and pressing the crawl button.

## Extending the crawler logic to be more flexible

Right now we specify the precise metadata location for any field we want to index. We should allow an auto-resolve feature to recursively crawl everything under some sub-path. 
  
### Example use cases.

1. *Index asset metadata indiscriminately*

Instead of doing this for every asset field:

synopsis: `site_map.searchables.public.asset_metadata.info.synopsis`

We could just do 

asset_tags: `public.*` or `public.asset_metadata.*` as `text` field

(If we remove the site object like I mentioned earlier, we can get rid of the site_map.searchables stuff)

2. *CRAWL EVERYTHING*

If we just want to "do" search we can do

content: `*` as `text` field

Everything will be searchable under one field caleld `content`

3. *Cache asset metadata information to include in results (already works)*

Suppose we are doing clip search, rather than querying the metadata again after doing the tantivy search (`select=` param), we can do this:

asset_info `public.asset_metadata.*` as `json` field

The json fields are not searchable like the `text` fields, but we can show them in the results with `display_fields=f_asset_info`.

So the field type controls how it is crawled & indexed, if we select `json` it will all be stored in one serialized json field. If we specify `text` we will recurisively crawl the asset_metadata and index all the leaf values. 

**Detail**: It's possible that paths can overlap. E.g. we could have `*` and `public.*.` which both crawl everything under public. I believe we should defer to the more specific path in the case of conflicts and not store the data twice in two separate fields. In the case that we have conflicting paths which don't have an ancestor/dependent relationship, we should throw an error. I.e if we have `public.*` and we have `*.asset_metadata` then there is no clear hierarchy between these paths and I think it's simplest to just throw an error. The crawl paths need to form a DAG. 

An example use case where this solution is a good idea is this:

speech_to_text `site_map.searchables.*.video_tags.metadata_tags.*.metadata_tags.shot_tags.tags.text.Speech to Text.text`

crawls all speech to text tags and puts them under the speech_to_text field

tags `site_map.searchables.*.video_tags.metadata_tags.*.metadata_tags.shot_tags.tags.text.*.text`

crawls all tags and puts them under the tags field

With the above compromise, we can specify a generic "tags" field that will crawl all of tags that have not been explicitly specified beforehand, and there won't be any weirdness where we store the tags twice. This way, if we add a new tags track, we have a reasonable default behavior without needing to reconfigure the index to make the crawler aware of the new field. 


## Unified API

The ai.contentfabric.io/search will be the go to API for users, but the fabric search will still exist and be used behind the scenes. 

### Update flow

1. User creates a write token and passes it to the search API
2. The search will run the bitcode "crawl" LRO on the fabric to update the index, it will then build the vector index and finalize the write token (not commit). By finalizing we get a hash, and we can store a reference to the internal vector-store by the hash.
3. Client commits the write token. If the user forgets this step nothing bad will happen since the vector search references indices by their version hash. 