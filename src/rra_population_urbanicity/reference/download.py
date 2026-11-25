"""Downloading reference files: streamed to disk, and unpacked where archived."""

import zipfile
from pathlib import Path

import requests
from rra_tools.shell_tools import mkdir

TIMEOUT_SECONDS = 60
CHUNK_BYTES = 2**20


def fetch(url: str, path: Path) -> None:
    """Stream a URL to a file, under a temporary name until it is complete."""
    partial = path.with_name(f"{path.name}.part")
    with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with partial.open("wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                f.write(chunk)
    partial.rename(path)


def download_archive(url: str, out_dir: Path, expected: str) -> Path:
    """Download and unpack a zip archive, unless the file expected from it is there.

    Everything in the archive is unpacked into `out_dir`, and the archive is
    then removed.
    """
    path = out_dir / expected
    if not path.exists():
        mkdir(out_dir, parents=True, exist_ok=True)
        archive = out_dir / url.rsplit("/", 1)[-1]
        fetch(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        archive.unlink()
    if not path.exists():
        msg = f"{url} did not hold {expected}."
        raise FileNotFoundError(msg)
    return path
