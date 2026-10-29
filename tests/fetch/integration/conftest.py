
import subprocess

import pytest

from src.common.content import Content
from src.fetch.factory import FetchFactory
from src.fetch.model import DownloadResult, FetchSession, FetcherConfig, MediaMetadata, Source

@pytest.fixture
def fetcher(fetcher_config: FetcherConfig, tag_store, qfactory) -> FetchFactory:
    """Create a FetchFactory instance for testing"""
    return FetchFactory(config=fetcher_config, ts=tag_store, qfactory=qfactory)

@pytest.fixture
def black_frame_fetcher(temp_dir) -> FetchSession:

    def generate_black_video(
        output_path: str,
        duration: float,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
    ) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"color=c=black:s={width}x{height}:r={fps}",
                "-t", str(duration),
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-y",
                output_path,
            ],
            check=True,
            capture_output=True,
        )

    class BlackFrameFetcher(FetchSession):
        def __init__(self, interval: int, duration: int, output_dir: str):
            self.interval = interval
            self.duration = duration
            self.output_dir = output_dir

        def download(self):
            res = []
            for i in range(self.duration // self.interval):
                filepath = f"{self.output_dir}/{i}.mp4"
                generate_black_video(output_path=filepath, duration=self.interval)
                res.append(
                    Source(
                        name=str(i),
                        filepath=filepath,
                        offset=i * self.interval,
                        wall_clock=None
                    )
                )
            # generate the last one
            if self.duration % self.interval != 0:
                i = self.duration // self.interval
                filepath = f"{self.output_dir}/{i}.mp4"
                generate_black_video(output_path=filepath, duration=self.duration % self.interval)
                res.append(
                    Source(
                        name=str(i),
                        filepath=filepath,
                        offset=i * self.interval * 1000, # convert to ms
                        wall_clock=None
                    )
                )


            return DownloadResult(sources=res, failed=[], done=True)

        def metadata(self) -> MediaMetadata:
            return MediaMetadata(
                sources=[f"{self.output_dir}/{j}.mp4" for j in range(self.duration // self.interval + (1 if self.duration % self.interval != 0 else 0))],
                fps=None
            )

        @property
        def path(self) -> str:
            return self.output_dir

    return BlackFrameFetcher(interval=10, duration=11, output_dir=temp_dir)

@pytest.fixture
def legacy_vod_content_with_tags_clean(q_legacy, tag_store) -> Content:

    batches = tag_store.find_batches(q=q_legacy)
    for batch in batches:
        tag_store.delete_batch(batch.id, q=q_legacy)

    return q_legacy


@pytest.fixture
def vod_content_with_tags_clean(q, tag_store) -> Content:

    batches = tag_store.find_batches(q=q)
    for batch in batches:
        tag_store.delete_batch(batch.id, q=q)

    return q

@pytest.fixture
def assets_content_with_tags_clean(q_assets, tag_store) -> Content:
    batches = tag_store.find_batches(q=q_assets)
    for batch in batches:
        tag_store.delete_batch(batch.id, q=q_assets)

    return q_assets