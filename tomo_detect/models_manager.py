"""Helpers to manage large model weight files outside the package distribution.

This module provides a minimal helper that checks for expected model files
in the package `tomo_detect/models/` directory and attempts to download them
from a user-provided base URL (environment variable) if they are missing.

Why: model weight files are large and should not be bundled into the PyPI
distribution. Instead, host them separately (GitHub Releases, S3, etc.) and
set the environment variable TOMO_DETECT_MODEL_BASE_URL to point to the
directory containing the files. Example:

  export TOMO_DETECT_MODEL_BASE_URL=https://example.com/tomo-detect-models/

The helper will try to download files like "r3d200_704_350984_epoch400.pt"
from: {base_url}/{filename}

If you prefer not to host models, instruct users to manually place the
required .pt files into the `tomo_detect/models/` directory.
"""
import os
import sys
import urllib.request
from pathlib import Path
import json
from typing import Dict, List, Optional

# List of model filenames that the package expects by default.
# If you add or remove models, update this list accordingly.
DEFAULT_MODELS: List[str] = [
    "r3d200_704_350984_epoch400.pt",
    "r3d200_704_360434_epoch400.pt",
    "r3d200_704_503766_epoch400.pt",
    "r3d200_704_601719_epoch400.pt",
    "r3d200_704_665991_epoch400.pt",
    "r3d200_704_684578_epoch400.pt",
    "r3d200_704_754606_epoch400.pt",
    "r3d200_704_911544_epoch400.pt",
]


def get_models_dir() -> Path:
    base = Path(__file__).resolve().parent
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def model_path(filename: str) -> Path:
    return get_models_dir() / filename


def _download_url(url: str, dest: Path) -> None:
    """Download url -> dest using urllib with a simple progress indicator."""
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100
        sys.stdout.write(f"\rDownloading {dest.name}: {pct:.1f}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, filename=str(dest), reporthook=_reporthook)
        sys.stdout.write("\r" + " " * 80 + "\r")
        print(f"Downloaded {dest.name}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url} -> {dest}: {e}") from e


def _load_manifest(manifest_path: Optional[Path] = None) -> Dict[str, str]:
    """Load a manifest mapping filenames -> URLs.

    Manifest location defaults to a file named 'models_manifest.json' next to
    this module. If no manifest exists, returns an empty dict.
    """
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parent / "models_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(mapping: Dict[str, str], manifest_path: Optional[Path] = None) -> None:
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parent / "models_manifest.json"
    with open(manifest_path, 'w', encoding='utf8') as f:
        json.dump(mapping, f, indent=2)


def register_model_url(filename: str, url: str, manifest_path: Optional[Path] = None) -> None:
    """Register a download URL for a model filename in the manifest file."""
    mapping = _load_manifest(manifest_path)
    mapping[filename] = url
    _save_manifest(mapping, manifest_path)


def register_from_drive_folder(folder_url: str, manifest_path: Optional[Path] = None) -> Dict[str, str]:
    """Attempt to discover files in a public Google Drive folder and register
    each file's share URL in the manifest.

    This does a best-effort HTML scrape of the folder page to find file ids
    and names. It requires the folder to be publicly accessible. If `requests`
    is not available in the environment, this function will raise an
    informative RuntimeError asking the user to install it.

    Returns the mapping of filename -> share_url that was written to the manifest.
    """
    try:
        import requests
    except Exception:
        raise RuntimeError("register_from_drive_folder requires the 'requests' package. Please install it and try again.")

    # Fetch folder HTML
    resp = requests.get(folder_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch folder URL: {folder_url} (status={resp.status_code})")

    html = resp.text

    # Heuristic: look for '/file/d/FILEID' patterns and the adjacent file name
    import re
    file_entries = {}

    # Find patterns like '/file/d/FILEID' and try to capture the file id
    for match in re.finditer(r"/file/d/([a-zA-Z0-9_-]{10,})", html):
        fid = match.group(1)
        # Build a shareable URL for the file
        share_url = f"https://drive.google.com/file/d/{fid}/view?usp=sharing"
        # Try to find the filename near the match in the HTML (look ahead/back)
        span_start = max(0, match.start() - 200)
        span_end = min(len(html), match.end() + 200)
        snippet = html[span_start:span_end]
        # Attempt to find a quoted filename inside the snippet
        name_match = re.search(r'"([^"]+\.(pt|pth|zip|tar|tar.gz|tgz))"', snippet, flags=re.IGNORECASE)
        if name_match:
            fname = name_match.group(1)
        else:
            # Fallback: use the file id as placeholder name
            fname = f"{fid}.pt"
        file_entries[fname] = share_url

    if not file_entries:
        raise RuntimeError("No files discovered in the provided Google Drive folder URL. Ensure the folder is public and contains the expected model files.")

    # Register all discovered entries in the manifest
    manifest = _load_manifest(manifest_path)
    manifest.update(file_entries)
    _save_manifest(manifest, manifest_path)
    return file_entries


def _download_from_google_drive(file_id: str, dest: Path) -> None:
    """Download a file from Google Drive given a shareable file id.

    This function prefers `requests` (for the confirm token handling used for
    large files). If `requests` is not available, falls back to a simple
    uc?export=download URL via urllib which may fail for large files.
    """
    # Prefer requests for robust streaming + confirm token handling
    try:
        import requests
    except Exception:
        # Fallback: try the uc?export=download URL (may not work for large files)
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        _download_url(url, dest)
        return

    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"

    response = session.get(URL, params={"id": file_id}, stream=True)

    # Check for confirm token in cookies
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def ensure_model(filename: str, base_url: str = None) -> Path:
    """Ensure the named model exists in the package models directory.

    Lookup order when attempting to download:
      1. models_manifest.json mapping (filename -> URL)
      2. TOMO_DETECT_MODEL_BASE_URL (base URL) + filename

    The manifest allows per-file URLs (e.g., Google Drive share links).
    """
    p = model_path(filename)
    if p.exists():
        return p

    # Try manifest first
    manifest = _load_manifest()
    if filename in manifest:
        url = manifest[filename]
        print(f"Found manifest URL for {filename}: {url}")
        # Special-case Google Drive links
        if "drive.google.com" in url:
            # extract file id if possible
            # common forms: https://drive.google.com/file/d/FILEID/view?usp=sharing
            # or https://drive.google.com/open?id=FILEID
            fid = None
            if "/d/" in url:
                parts = url.split("/d/")
                if len(parts) > 1:
                    fid = parts[1].split("/")[0]
            elif "id=" in url:
                fid = url.split("id=")[-1].split("&")[0]
            if fid:
                _download_from_google_drive(fid, p)
            else:
                # fall back to direct URL download
                _download_url(url, p)
        else:
            _download_url(url, p)
        if not p.exists():
            raise RuntimeError(f"Downloaded file not found after download: {p}")
        return p

    # fallback to base_url env var
    env_url = os.environ.get("TOMO_DETECT_MODEL_BASE_URL")
    base = base_url or env_url
    if not base:
        raise RuntimeError(
            f"Model '{filename}' not found at {p}.\n"
            "Set the environment variable TOMO_DETECT_MODEL_BASE_URL to a URL where\n"
            "the pretrained model files are hosted, or manually place the file\n"
            f"at: {p}\n"
            "Alternatively, register a per-file URL using models_manager.register_model_url(filename, url)."
        )

    # ensure base ends with '/'
    if not base.endswith("/"):
        base = base + "/"

    url = base + filename
    print(f"Model {filename} not found locally. Attempting download from: {url}")
    _download_url(url, p)
    if not p.exists():
        raise RuntimeError(f"Downloaded file not found after download: {p}")
    return p

def ensure_all_models(filenames: List[str] = None, base_url: str = None) -> Dict[str, Path]:
    """Ensure all models in filenames exist; try downloading missing ones.

    Returns a dict mapping filename -> Path.
    """
    filenames = filenames or DEFAULT_MODELS
    out = {}
    for fn in filenames:
        out[fn] = ensure_model(fn, base_url=base_url)
    return out
