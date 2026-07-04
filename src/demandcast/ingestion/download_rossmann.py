"""Download the Rossmann Store Sales dataset from Kaggle into data/raw.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (Account -> Create New
Token on kaggle.com). The kaggle package reads credentials from there or
from the KAGGLE_USERNAME / KAGGLE_KEY environment variables.
"""

import logging
import zipfile
from pathlib import Path

from demandcast.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download(raw_dir: Path, kaggle_slug: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading dataset '%s' to %s", kaggle_slug, raw_dir)
    api.dataset_download_files(kaggle_slug, path=str(raw_dir), unzip=False, quiet=False)

    zip_path = raw_dir / f"{kaggle_slug.split('/')[-1]}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()


def verify(raw_dir: Path, required_files: list[str]) -> None:
    missing = [f for f in required_files if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Expected files missing from {raw_dir} after download: {missing}")
    logger.info("Verified presence of %s in %s", required_files, raw_dir)


def main() -> None:
    config = load_config()
    download(config.dataset.raw_dir, config.dataset.kaggle_slug)
    verify(config.dataset.raw_dir, config.dataset.required_files)


if __name__ == "__main__":
    main()
