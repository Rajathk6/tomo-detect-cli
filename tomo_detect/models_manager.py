"""
Helpers to manage large model weight files outside the package distribution.

This module ensures that required pretrained model files are available in the
`tomo_detect/models/` directory. If a model file is missing, it attempts to
download it automatically from a predefined Google Cloud Storage (GCS) bucket.

If the Google Cloud links do not work, manually download the models from Google
Drive. Refer to the README for manual download instructions.

Environment variable option:
  export TOMO_DETECT_MODEL_BASE_URL=https://storage.googleapis.com/pre-trained-model-tomo-detect/

Alternatively, the `models_manifest.json` file defines explicit URLs for each
model. By default, the manifest points to the GCS bucket URLs.
"""

import os
import sys
import urllib.request
import json
from pathlib import Path
from typing import Dict, List, Optional

# Default single model for testing
DEFAULT_MODELS: List[str] = [
    "r3d200_704_350984_epoch400.pt",
]

# Default manifest mapping to Google Cloud bucket
DEFAULT_MANIFEST: Dict[str, str] = {
    "r3d200_704_350984_epoch400.pt": "https://storage.googleapis.com/pre-trained-model-tomo-detect/r3d200_704_350984_epoch400.pt",
    "r3d200_704_360434_epoch400.pt": "https://storage.googleapis.com/pre-trained-model-tomo-detect/r3d200_704_360434_epoch400.pt",
    "r3d200_704_503766_epoch400.pt": "https://storage.googleapis.com/pre-trained-model-tomo-detect/r3d200_704_503766_epoch400.pt",
    "r3d200_704_601719_epoch400.pt": "https://storage.googleapis.com/pre-trained-model-tomo-detect/r3d200_704_601719_epoch400.pt",
    "r3d200_704_665991_epoch400.pt": "https://storage.googleapis.com/pre-trained-model-tomo-detect/r3d200_704_665991_epoch400.pt",
    "r3d200_704_684578_epoch400.pt": "https://storage.googleapis.com/pre-trained-model-tomo-detect/r3d200_704_684578_epoch400.pt",
    "r3d200_704_754606_epoch400.pt": "https://storage.googleapis.com/pre-trained-model-tomo-detect/r3d200_704_754606_epoch400.pt",
    "r3d200_704_911544_epoch400.pt": "https://storage.googleapis.com/pre-trained-model-tomo-detect/r3d200_704_911544_epoch400.pt",
}


def get_models_dir() -> Path:
    """Return the local path where models are stored, creating it if needed."""
    base = Path(__file__).resolve().parent
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def model_path(filename: str) -> Path:
    """Return the full local path to a model file."""
    return get_models_dir() / filename


def _download_url(url: str, dest: Path) -> None:
    """Download a file from the given URL to destination."""
    dest.parent.mkdir(parents=True, exist_ok=True)

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
    """Load model manifest or return default mapping."""
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parent / "models_manifest.json"

    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf8') as f:
                return json.load(f)
        except Exception:
            return DEFAULT_MANIFEST.copy()
    return DEFAULT_MANIFEST.copy()


def _save_manifest(mapping: Dict[str, str], manifest_path: Optional[Path] = None) -> None:
    """Save model manifest to file."""
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parent / "models_manifest.json"
    with open(manifest_path, 'w', encoding='utf8') as f:
        json.dump(mapping, f, indent=2)


def register_model_url(filename: str, url: str, manifest_path: Optional[Path] = None) -> None:
    """Register or update a single model URL in the manifest."""
    mapping = _load_manifest(manifest_path)
    mapping[filename] = url
    _save_manifest(mapping, manifest_path)


def ensure_model(filename: str, base_url: str = None) -> Path:
    """Ensure that a specific model file exists locally, downloading if missing."""
    p = model_path(filename)
    if p.exists():
        return p

    manifest = _load_manifest()
    url = manifest.get(filename)
    if url:
        print(f"Model {filename} not found locally. Downloading from manifest URL:\n{url}")
        _download_url(url, p)
        if not p.exists():
            raise RuntimeError(f"Downloaded file not found after download: {p}")
        return p

    # fallback to env var
    env_url = os.environ.get("TOMO_DETECT_MODEL_BASE_URL")
    base = base_url or env_url
    if not base:
        raise RuntimeError(
            f"Model '{filename}' not found at {p}.\n"
            "Set the environment variable TOMO_DETECT_MODEL_BASE_URL to a valid URL,\n"
            "or manually place the model file in the 'tomo_detect/models/' directory.\n"
            "If the Google Cloud link fails, refer to the README for Google Drive download instructions."
        )

    if not base.endswith("/"):
        base += "/"

    url = base + filename
    print(f"Model {filename} not found locally. Attempting download from: {url}")
    _download_url(url, p)
    if not p.exists():
        raise RuntimeError(f"Downloaded file not found after download: {p}")
    return p


def ensure_all_models(filenames: List[str] = None, base_url: str = None) -> Dict[str, Path]:
    """Ensure all required models exist locally. For testing, defaults to one model."""
    filenames = filenames or DEFAULT_MODELS
    out = {}
    for fn in filenames:
        out[fn] = ensure_model(fn, base_url=base_url)
    return out


if __name__ == "__main__":
    """Quick test mode. Downloads one model file if missing."""
    print("Running model manager test...")
    ensure_all_models()
    print("Test complete. Model(s) are available in 'tomo_detect/models/'.")
