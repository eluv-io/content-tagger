
from loguru import logger

from src.common.content import Content
from src.tags.tagstore.tagstore import FilesystemTagStore
from src.tags.conversion import TagConverter

def commit(self, q: Content, converter: TagConverter, tagstore: FilesystemTagStore) -> None:
    """
    Format and upload tags for a content object using tags stored in the tagstore.
    
    Args:
        q: Content object with write token
        interval: Interval in minutes to bucket the formatted results (default: 10)
    """
    logger.info(f"Starting commit for content {q.qhit}")
    
    # Get tags for this content
    tags = self._get_latest_tags_for_content(q.qhit)
    if not tags:
        logger.info(f"No tags found for content {q.qhit}")
        return

    # Parse tags into video and frame formats
    all_video_tags, all_frame_tags = self._parse_tags(tags)
    
    if not all_video_tags:
        logger.info(f"No tags found for content {q.qhit}")
        return
    
    # Process tags
    formatted_tracks, overlays = self._format_tags_for_upload(
        all_video_tags, all_frame_tags, interval
    )
    
    # Upload formatted tags
    self._upload_formatted_tags(q, formatted_tracks, overlays)
    
    logger.info(f"Successfully committed tags for content {q.qhit}")

def _add_links(self, q: Content, filenames: list[str]) -> None:
    """Add metadata links for uploaded tag files."""
    data = {}
    
    for filename in filenames:
        if 'video-tags-tracks' in filename:
            tag_type = 'metadata_tags'
        elif 'video-tags-overlay' in filename:
            tag_type = 'overlay_tags'
        else:
            continue
        
        if tag_type not in data:
            data[tag_type] = {}

        idx = ''.join([char for char in filename if char.isdigit()])
        data[tag_type][idx] = {"/": f"./files/video_tags/{filename}"}
    
    if data:
        q.merge_metadata(metadata=data, metadata_subtree='video_tags')