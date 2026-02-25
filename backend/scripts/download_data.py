"""
CLI script to download Google Cluster Data.

Usage:
    python scripts/download_data.py                    # Default 3 shards
    python scripts/download_data.py --shards 5         # 5 shards
    python scripts/download_data.py --synthetic        # Generate synthetic only
"""
import argparse
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DataConfig, DATA_RAW, DATA_PROCESSED
from app.data.downloader import GoogleClusterDownloader
from app.data.loader import ClusterDataLoader
from app.data.preprocessor import Preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download & prepare GPU forecast data")
    parser.add_argument("--shards", type=int, default=3, help="Number of shards to download")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead")
    parser.add_argument("--process", action="store_true", help="Also run preprocessing")
    parser.add_argument("--machines", type=int, default=8, help="Number of top machines to keep")
    args = parser.parse_args()

    config = DataConfig(max_shards_per_table=args.shards)

    if args.synthetic:
        # ── Synthetic data (no download needed) ──
        logger.info("Generating synthetic GPU telemetry data...")
        machines = ClusterDataLoader.generate_synthetic(config)
        logger.info(f"Generated {len(machines)} synthetic GPUs")

    else:
        # ── Download real Google Cluster Data ──
        logger.info("Downloading Google Cluster Data 2019...")
        downloader = GoogleClusterDownloader(config)
        csv_paths = downloader.download_instance_usage(max_shards=args.shards)

        if not csv_paths:
            logger.error("No data downloaded. Check your network connection.")
            sys.exit(1)

        logger.info("Loading downloaded data...")
        loader = ClusterDataLoader(config)
        raw_df = loader.load_instance_usage(csv_paths)
        machines = loader.load_per_machine(raw_df, top_n_machines=args.machines)

    if args.process:
        # ── Preprocess into model-ready arrays ──
        logger.info("Running preprocessing pipeline...")
        preprocessor = Preprocessor(config)
        all_splits = preprocessor.process_all_machines(machines)

        # Save processed data
        import numpy as np
        for mid, splits in all_splits.items():
            out_dir = DATA_PROCESSED / f"machine_{mid}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
                np.save(out_dir / f"{key}.npy", splits[key])
            logger.info(f"Saved processed data for machine {mid}")

        preprocessor.save_scalers()
        logger.info("Preprocessing complete!")

    logger.info("Done!")


if __name__ == "__main__":
    main()