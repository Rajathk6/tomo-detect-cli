import os
import sys
import platform
import psutil
import logging
from pathlib import Path
import numpy as np
import mrcfile
from rich.console import Console
from rich.logging import RichHandler

console = Console()

def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(
            rich_tracebacks=True,
            console=console,
            show_time=False
        )]
    )

def print_system_info() -> None:
    """Print detailed system information"""
    console.print("\n[bold blue]System Information:[/]")
    
    # OS Info
    os_info = platform.platform()
    console.print(f"[cyan]OS:[/] {os_info}")
    
    # CPU Info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    console.print(f"[cyan]CPU:[/] {cpu_count} cores, {cpu_threads} threads")
    
    # Memory Info
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024 ** 3)
    console.print(f"[cyan]RAM:[/] {ram_gb:.1f} GB total")
    
    # GPU Info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"[cyan]GPU:[/] {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            console.print("[cyan]GPU:[/] Not available")
    except:
        console.print("[cyan]GPU:[/] Error detecting GPU")
    
    # Python Info
    py_version = platform.python_version()
    console.print(f"[cyan]Python:[/] {py_version}")
    
    console.print("") # Empty line for spacing

def load_tomo_stack(image_files):
    """Load a stack of tomography images into a 3D numpy array"""
    import PIL.Image as Image
    images = []
    for img_file in sorted(image_files):
        with Image.open(img_file) as img:
            images.append(np.array(img))
    return np.stack(images)

def validate_input_file(file_path: Path, preserve_original_range: bool = False) -> np.ndarray:
    """Validate and load input file.

    Parameters
    - file_path: Path to input file (.npy, .mrc, .zip, or image files) or directory containing image slices
    - preserve_original_range: when True, do not rescale the array to [0,1]
      even if values appear outside the [0,1] range. Many models expect
      raw 0-255 input and perform division by 255 internally; use this
      flag to preserve that behavior.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Handle directory of images
    if file_path.is_dir():
        image_files = []
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            image_files.extend(file_path.glob(f'*{ext}'))
            image_files.extend(file_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No image files found in directory: {file_path}")
            
        console.print(f"Loading {len(image_files)} image files from directory...")
        data = load_tomo_stack(sorted(image_files))
        console.print(f"Loaded image stack with shape: {data.shape}")
        return data

    suffix = file_path.suffix.lower()

    # Load based on file type
    if suffix == '.npy':
        data = np.load(file_path)

    elif suffix == '.mrc':
        with mrcfile.open(file_path) as mrc:
            data = mrc.data

    elif suffix == '.zip':
        import zipfile
        import tempfile

        # Create temp directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Check for .npy or .mrc files first
                npy_mrc_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.npy', '.mrc'))]
                if npy_mrc_files:
                    # Extract first .npy or .mrc file found and recurse
                    target_file = npy_mrc_files[0]
                    zip_ref.extract(target_file, temp_dir)
                    return validate_input_file(Path(temp_dir) / target_file, preserve_original_range=preserve_original_range)

                # If no .npy/.mrc files, look for image files
                image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
                if not image_files:
                    raise ValueError("No supported files found in zip archive")

                # Extract all images and load stack
                zip_ref.extractall(temp_dir)
                image_paths = [Path(temp_dir) / f for f in image_files]
                data = load_tomo_stack(image_paths)

    else:
        # Check if it's an image file
        if suffix in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
            import PIL.Image as Image
            data = np.array(Image.open(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # Validate dimensions
    if len(data.shape) != 3:
        raise ValueError(
            f"Expected 3D array, got {len(data.shape)}D. Shape: {data.shape}"
        )

    # Validate data type
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    # Normalize only when requested. By default the function will
    # normalize to [0,1] if values appear outside that range. When
    # preserve_original_range=True the original numeric range is left
    # untouched (useful when the model itself performs the final
    # division, e.g. `x = x / 255.0` inside the network).
    if not preserve_original_range:
        if data.max() > 1.0 or data.min() < 0.0:
            data = (data - data.min()) / (data.max() - data.min())

    return data