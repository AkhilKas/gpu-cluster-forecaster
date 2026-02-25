"""
Download Google Cluster Data 2019 from GCS.

The dataset lives at gs://clusterdata_2019_a/ and contains:
  - instance_usage/    → CPU & memory usage per task instance (OUR MAIN TABLE)
  - instance_events/   → task lifecycle events
  - machine_events/    → machine add/remove/update events
  - collection_events/ → job-level metadata

Each table is sharded into ~500 CSV.gz files.
We download a configurable number of shards to keep things manageable.

Docs: https://github.com/google/cluster-data/blob/master/ClusterData2019.md
"""

import gzip
import logging
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.config import DATA_RAW, DataConfig

logger = logging.getLogger(__name__)

GCS_BASE_URL = "https://storage.googleapis.com/clusterdata_2019_a"


class GoogleClusterDownloader:
    """Downloads and extracts Google Cluster Data 2019 shards."""

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()
        self.raw_dir = DATA_RAW / "google_cluster_2019"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _shard_url(self, table: str, shard_idx: int) -> str:
        """Construct GCS URL for a specific table shard."""
        # Files are named like: instance_usage-000000000000.csv.gz
        return f"{GCS_BASE_URL}/{table}/{table}-{shard_idx:012d}.csv.gz"

    def _download_shard(self, table: str, shard_idx: int) -> Path | None:
        """Download a single shard if not already cached."""
        table_dir = self.raw_dir / table
        table_dir.mkdir(exist_ok=True)

        gz_path = table_dir / f"{table}-{shard_idx:012d}.csv.gz"
        csv_path = table_dir / f"{table}-{shard_idx:012d}.csv"

        # Skip if already downloaded
        if csv_path.exists():
            logger.debug(f"Already exists: {csv_path.name}")
            return csv_path

        url = self._shard_url(table, shard_idx)
        logger.info(f"Downloading {url}")

        try:
            urllib.request.urlretrieve(url, gz_path)

            # Decompress
            with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            gz_path.unlink()  # Remove .gz after extraction
            logger.info(f"Extracted: {csv_path.name}")
            return csv_path

        except Exception as e:
            logger.error(f"Failed to download shard {shard_idx} of {table}: {e}")
            # Clean up partial downloads
            for p in [gz_path, csv_path]:
                if p.exists():
                    p.unlink()
            return None

    def download(
        self, tables: list | None = None, max_shards: int | None = None
    ) -> dict:
        """
        Download multiple shards for specified tables.

        Returns:
            dict mapping table name -> list of downloaded CSV paths
        """
        tables = tables or self.config.tables
        max_shards = max_shards or self.config.max_shards_per_table
        downloaded = {}

        for table in tables:
            logger.info(f"Downloading {max_shards} shards of '{table}'...")
            paths = []

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._download_shard, table, i): i
                    for i in range(max_shards)
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        paths.append(result)

            paths.sort()
            downloaded[table] = paths
            logger.info(f"  -> {len(paths)} shards downloaded for '{table}'")

        return downloaded

    def download_instance_usage(self, max_shards: int | None = None) -> list:
        """Convenience: download only instance_usage (the main table we need)."""
        result = self.download(
            tables=["instance_usage"],
            max_shards=max_shards or self.config.max_shards_per_table,
        )
        return result.get("instance_usage", [])
