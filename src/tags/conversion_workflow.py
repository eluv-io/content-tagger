import os
import json
import tempfile
import shutil
from loguru import logger
from elv_client_py import ElvClient
from common_ml.utils.metrics import timeit
from src.common.content import Content
from src.tags.tagstore.tagstore import FilesystemTagStore
from src.tags.conversion import TagConverter
from src.tags.conversion import get_latest_tags_for_content

def upload_tags_to_fabric(
    source_qhit: str,
    qwt: Content, 
    tagstore: FilesystemTagStore, 
    tag_converter: TagConverter
) -> None:
    """
    Complete workflow to convert tagstore tags to fabric format and upload them.
    
    Args:
        source_qhit: object qhit whose tags to use
        qwt: Write token to upload to
        tagstore: FilesystemTagStore instance
        converter_config: Configuration for tag conversion
    """
    logger.info(f"Starting tag upload for content {source_qhit}")

    # Step 1: Extract tags and jobs from tagstore
    logger.info("Extracting latest tags from tagstore")
    job_tags = get_latest_tags_for_content(source_qhit, tagstore)
    
    if not job_tags:
        logger.warning(f"No tags found for content {source_qhit}")
        return
    
    logger.info(f"Found {len(job_tags)} jobs with tags")
    
    # Step 3: Split into time-based buckets
    bucketed_job_tags = tag_converter.split_tags(job_tags)
    
    logger.info(f"Created {len(bucketed_job_tags)} time buckets")
    
    # Step 4: Convert each bucket and prepare upload files
    to_upload = []

    tags_dir = tempfile.mkdtemp()

    # Process tracks (metadata tags)
    for i, bucket in enumerate(bucketed_job_tags):
        if not any(jt.tags for jt in bucket):  # Skip empty buckets
            continue
            
        logger.debug(f"Processing track bucket {i}")
        
        # Convert to TrackCollection and dump
        track_collection = tag_converter.get_tracks(bucket)
        track_json = tag_converter.dump_tracks(track_collection)

        # Write to file
        track_filename = f"video-tags-tracks-{i:04d}.json"
        track_filepath = os.path.join(tags_dir, track_filename)
        with open(track_filepath, 'w') as f:
            json.dump(track_json, f)
        
        to_upload.append(track_filepath)
        logger.debug(f"Created track file: {track_filename}")
    
    # Process overlays (frame-level tags)
    for i, bucket in enumerate(bucketed_job_tags):
        if not any(jt.tags for jt in bucket):  # Skip empty buckets
            continue
            
        logger.debug(f"Processing overlay bucket {i}")
        
        # Convert to Overlay and dump
        overlay = tag_converter.get_overlays(bucket)

        if not overlay:  # Skip if no frame-level tags
            continue

        overlay_json = tag_converter.dump_overlay(overlay)
        
        # Write to file
        overlay_filename = f"video-tags-overlay-{i:04d}.json"
        overlay_filepath = os.path.join(tags_dir, overlay_filename)

        with open(overlay_filepath, 'w') as f:
            json.dump(overlay_json, f)
        
        to_upload.append(overlay_filepath)
        logger.debug(f"Created overlay file: {overlay_filename}")
    
    if not to_upload:
        logger.warning("No tag files to upload")
        return
    
    logger.info(f"Prepared {len(to_upload)} files for upload")
    
    # Step 5: Upload files to fabric
    logger.info("Uploading tag files to fabric")
    
    file_jobs = [
        ElvClient.FileJob(
            local_path=filepath, 
            out_path=f"video_tags/{os.path.basename(filepath)}", 
            mime_type="application/json"
        ) 
        for filepath in to_upload
    ]
   
    with timeit("Uploading tag files"):
        qwt.upload_files(file_jobs=file_jobs, finalize=False)
    
    logger.info("Tag files uploaded successfully")
    
    # Step 6: Add metadata links
    logger.info("Adding metadata links")
    
    with timeit("Adding metadata links"):
        _add_tag_links(qwt, to_upload)

    logger.info(f"Tag upload completed for content {source_qhit}")

    shutil.rmtree(tags_dir)

def _add_tag_links(qwt: Content, filepaths: list[str]) -> None:
    """
    Add metadata links for uploaded tag files.
    
    Args:
        qwt: Content object with write token
        filepaths: List of local file paths that were uploaded
    """
    metadata = {}
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        
        if 'video-tags-tracks' in filename:
            tag_type = 'metadata_tags'
        elif 'video-tags-overlay' in filename:
            tag_type = 'overlay_tags'
        else:
            logger.warning(f"Unknown tag file type: {filename}")
            continue
        
        if tag_type not in metadata:
            metadata[tag_type] = {}
        
        # Extract index from filename (e.g., "0001" from "video-tags-tracks-0001.json")
        idx = ''.join([char for char in filename if char.isdigit()])
        
        metadata[tag_type][idx] = {"/": f"./files/video_tags/{filename}"}
        logger.debug(f"Added link for {tag_type}[{idx}] -> {filename}")
    
    if metadata:
        qwt.merge_metadata(metadata=metadata, metadata_subtree='video_tags')
        logger.info(f"Added {len(metadata)} metadata link categories")