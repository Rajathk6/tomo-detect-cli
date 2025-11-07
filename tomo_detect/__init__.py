"""
Tomo-Detect: Motor coordinate detection in tomography data
======================================================

A command-line tool and library for detecting motor coordinates in tomography data 
using deep learning models.

Main Components:
--------------
- CLI interface for easy command-line usage
- StandaloneNet for deep learning model
- Support for .npy and .mrc file formats
- Built-in visualization and validation tools
"""

import warnings

# Filter out the specific PyTorch indexing warnings
warnings.filterwarnings('ignore', category=UserWarning, 
                       message='Using a non-tuple sequence for multidimensional indexing is deprecated*')

__version__ = "0.1.6"
__author__ = "Rajath Kumar"


# Avoid importing heavy submodules at package import time. Importing
# the package should be fast and not execute deep model imports — the
# CLI and consumers will request model-related symbols lazily when
# they're actually needed.
def __getattr__(name: str):
    """Lazily import selected attributes from submodules.

    This keeps package import lightweight and provides the same
    attribute names for backwards compatibility.
    """
    if name == "StandaloneNet":
        from .inference import StandaloneNet
        return StandaloneNet
    if name == "load_model":
        from .inference import load_model
        return load_model
    if name == "process_predictions":
        from .postprocess import process_predictions
        return process_predictions
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + ["StandaloneNet", "load_model", "process_predictions"]) 

__all__ = ["StandaloneNet", "load_model", "process_predictions"]