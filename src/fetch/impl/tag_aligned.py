
import os
import subprocess
import tempfile
from functools import lru_cache

from loguru import logger

from src.fetch.impl.vod import VodWorker
from src.fetch.model import DownloadResult, FetchSession, MediaMetadata, Source
from src.tags.reader.abstract import TagReader
from src.tags.tagstore.model import Tag

logger = logger.bind(module="fetch tag_aligned")


class TagAlignedFetcher(FetchSession):
    def __init__(
        self, 
        tr: TagReader,
        vod: VodWorker
    ):
        self.tr = tr
        self.fetcher = vod
        self._completed_tag_ids: set[str] = set()

    @property
    def path(self) -> str:
        return self.fetcher.path

    def metadata(self) -> MediaMetadata:
        meta = self.fetcher.metadata()
        tag_ids = [t.id for t in self._get_tags()]
        return MediaMetadata(
            fps=meta.fps,
            sources=tag_ids
        )
    
    def download(self) -> DownloadResult:
        vod_result = self.fetcher.download()

        if not vod_result.sources:
            return DownloadResult(
                sources=[],
                failed=vod_result.failed,
                done=vod_result.done,
            )

        # Build a time-sorted list of available media segments
        segments = sorted(vod_result.sources, key=lambda s: s.offset)

        tags = self._get_tags()
        output_dir = os.path.join(self.fetcher.path, "tag_aligned")
        os.makedirs(output_dir, exist_ok=True)

        result_sources: list[Source] = []
        failed: list[str] = list(vod_result.failed)

        for tag in tags:
            if tag.id in self._completed_tag_ids:
                continue

            tag_start_ms = tag.start_time
            tag_end_ms = tag.end_time

            # Find segments that overlap with the tag's time range
            overlapping = self._find_overlapping_segments(segments, tag_start_ms, tag_end_ms)
            if not overlapping:
                continue

            # Check if we have full coverage for this tag
            # (the available segments must span the entire tag range)
            coverage_start = overlapping[0].offset
            last_seg = overlapping[-1]
            # Estimate segment duration from part spacing or use the gap to next
            seg_duration_ms = self._estimate_segment_duration_ms(segments, last_seg)
            coverage_end = last_seg.offset + seg_duration_ms

            if coverage_start > tag_start_ms or coverage_end < tag_end_ms:
                # We don't have full coverage yet; skip for now unless vod is done
                if not vod_result.done:
                    continue

            ext = self._detect_extension(overlapping[0].filepath)
            out_path = os.path.join(output_dir, f"{tag.id}{ext}")

            try:
                self._extract_tag_segment(
                    overlapping, tag_start_ms, tag_end_ms, out_path, ext
                )
                result_sources.append(Source(
                    name=tag.id,
                    filepath=out_path,
                    offset=tag_start_ms,
                    wall_clock=None,
                ))
                self._completed_tag_ids.add(tag.id)
            except Exception as e:
                logger.error(f"Failed to extract segment for tag {tag.id}: {e}")
                failed.append(tag.id)

        return DownloadResult(
            sources=result_sources,
            failed=failed,
            done=vod_result.done,
        )

    @lru_cache(maxsize=1)
    def _get_tags(self) -> list[Tag]:
        return self.tr.read()

    @staticmethod
    def _find_overlapping_segments(
        segments: list[Source], start_ms: int, end_ms: int
    ) -> list[Source]:
        """Return segments whose time span overlaps [start_ms, end_ms)."""
        result = []
        for i, seg in enumerate(segments):
            seg_start = seg.offset
            # Estimate end from next segment or assume same duration as gap
            if i + 1 < len(segments):
                seg_end = segments[i + 1].offset
            else:
                # Last segment: estimate using the gap from the previous one
                if i > 0:
                    seg_end = seg.offset + (seg.offset - segments[i - 1].offset)
                else:
                    # Single segment, can't estimate duration well; include it
                    seg_end = seg.offset + (end_ms - start_ms)

            if seg_start < end_ms and seg_end > start_ms:
                result.append(seg)
        return result

    @staticmethod
    def _estimate_segment_duration_ms(segments: list[Source], seg: Source) -> int:
        """Estimate the duration of a segment in ms."""
        for i, s in enumerate(segments):
            if s is seg and i + 1 < len(segments):
                return segments[i + 1].offset - s.offset
        # Fallback: use gap from previous segment or a default
        for i, s in enumerate(segments):
            if s is seg and i > 0:
                return s.offset - segments[i - 1].offset
        # Single segment; return a large value
        return 60 * 60 * 1000

    @staticmethod
    def _detect_extension(filepath: str) -> str:
        _, ext = os.path.splitext(filepath)
        return ext if ext else ".mp4"

    @staticmethod
    def _extract_tag_segment(
        segments: list[Source],
        start_ms: int,
        end_ms: int,
        output_path: str,
        ext: str,
    ) -> None:
        """
        Use ffmpeg to concatenate and trim the overlapping segments to produce
        a media file that exactly covers [start_ms, end_ms).
        """
        if os.path.exists(output_path):
            return

        with tempfile.TemporaryDirectory(dir=os.path.dirname(output_path)) as tmp_dir:
            if len(segments) == 1:
                input_path = segments[0].filepath
                seg_offset = segments[0].offset
            else:
                # Concatenate segments first
                concat_list = os.path.join(tmp_dir, "concat.txt")
                with open(concat_list, "w") as f:
                    for seg in segments:
                        f.write(f"file '{seg.filepath}'\n")
                input_path = os.path.join(tmp_dir, f"concat{ext}")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", concat_list,
                        "-y",
                        "-c", "copy",
                        input_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                seg_offset = segments[0].offset

            # Trim to the exact tag time range
            ss = (start_ms - seg_offset) / 1000.0
            duration = (end_ms - start_ms) / 1000.0

            subprocess.run(
                [
                    "ffmpeg",
                    "-ss", f"{ss:.3f}",
                    "-i", input_path,
                    "-t", f"{duration:.3f}",
                    "-y",
                    "-map", "0",
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )