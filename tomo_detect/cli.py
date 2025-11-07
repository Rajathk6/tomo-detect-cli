import os
import sys
import json
import logging
import platform
import psutil
import zipfile
import mrcfile
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn
from typing import Union, List, Dict

import torch
import torch.nn.functional as F
from torch.amp import autocast
from monai.inferers import sliding_window_inference
import numpy as np
import pandas as pd

# Delay importing heavy inference/postprocess code until it's actually needed
# so importing the CLI module stays lightweight and package import won't
# fail when optional dependencies are missing. The functions will be
# imported locally inside process_input_file().
from .utils import setup_logging, print_system_info, validate_input_file

console = Console()

def process_input_file(file_path: Union[str, Path], 
                      output_dir: Union[str, Path],
                      batch_size: int = 1,
                      device: str = None,
                      debug: bool = False,
                      test: bool = False,
                      preserve_range: bool = True,
                      args = None) -> Dict:
    """Process a single input file and return results"""
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print file info
    console.print(f"\n[bold blue]Processing file:[/] {file_path.name}")
    console.print(f"[bold blue]Output directory:[/] {output_dir}")

    # Simply read and save the input file in both formats
    if file_path.suffix.lower() == '.npy':
        raw_data = np.load(file_path)
    elif file_path.suffix.lower() == '.mrc':
        with mrcfile.open(file_path) as mrc:
            raw_data = mrc.data

    # Save in both formats
    np.save(output_dir / f"{file_path.stem}_raw.npy", raw_data)
    with mrcfile.new(output_dir / f"{file_path.stem}_raw.mrc", overwrite=True) as mrc:
        mrc.set_data(raw_data.astype(np.float32))
    
    console.print("[green]✓[/] Saved raw input files")
    
    # Now proceed with validation and processing
    data = validate_input_file(file_path, preserve_original_range=preserve_range)
    console.print(f"[green]✓[/] Input validated - Shape: {data.shape}")
    
    # Helpful debug information about numeric range
    if debug:
        console.print(f"[yellow]Input range:[/] min={float(data.min()):.6f}, max={float(data.max()):.6f}")
    
    # Load model(s) lazily — wrap imports to provide clear error messages
    try:
        from .inference import load_model
    except Exception as e:
        console.print(f"\n[bold red]Failed to import inference module:[/] {e}")
        raise

    try:
        from .postprocess import process_predictions
    except Exception as e:
        console.print(f"\n[bold red]Failed to import postprocessing module:[/] {e}")
        raise

    model_dir = Path(__file__).parent / 'models'
    # If models are not included in the package distribution, attempt to
    # download them from a host specified by TOMO_DETECT_MODEL_BASE_URL.
    try:
        from . import models_manager
        # Ensure models directory exists
        model_dir.mkdir(parents=True, exist_ok=True)
        # If no models present, try to download the defaults
        present = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
        if not present:
            console.print("[yellow]No model weights found locally. Attempting to download default models...[/]")
            try:
                models_manager.ensure_all_models()
            except Exception as err:
                console.print("[red]Failed to download models automatically:[/] " + str(err))
                console.print("Please download model files and place them in: {}".format(model_dir))
        model_paths = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
    except Exception:
        # If models_manager fails for any reason, fall back to listing local models only
        model_paths = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
    models = []  # list of dicts: {'model': model_obj, 'cfg': optional_cfg}

    console.print("\n[bold blue]Loading models...")
    
    # If in test mode, only use the first model
    if test:
        model_paths = model_paths[:1]
        console.print("[yellow]Test mode:[/] Using only one model for faster testing")
    
    for mpath in model_paths:
        fpath = model_dir / mpath
        loaded = load_model(fpath, device)
        # Support both APIs: load_model may return model or (model, cfg)
        if isinstance(loaded, tuple):
            model_obj, model_cfg = loaded
        else:
            model_obj, model_cfg = loaded, None
        models.append({"model": model_obj, "cfg": model_cfg})
    
    # Run inference
    console.print("\n[bold blue]Running inference...")
    volume_tensor = torch.from_numpy(data).float()
    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
    volume_tensor = volume_tensor.to(device)

    # Default patch size used by the models
    default_patch_size = (128, 512, 512)

    preds = None
    with torch.no_grad(), autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        for idx, row in enumerate(models):
            model_obj = row.get("model")
            model_cfg = row.get("cfg")

            console.print(f"Running model {idx+1}/{len(models)}")

            # Determine per-model ROI / overlap / sw_batch_size if available
            roi_size = getattr(model_cfg, 'roi_size', default_patch_size) if model_cfg is not None else default_patch_size
            sw_batch = getattr(model_cfg, 'infer_cfg', None)
            sw_batch_size = getattr(sw_batch, 'sw_batch_size', batch_size) if sw_batch is not None else batch_size
            overlap = getattr(sw_batch, 'overlap', 0.5) if sw_batch is not None else 0.5

            # Predictor that calls model once and handles dict / tensor return types
            def _predictor(x, m=model_obj, out_size=roi_size):
                out = m(x)
                if isinstance(out, dict):
                    out = out.get('logits', out)
                return F.interpolate(out, size=out_size, mode='trilinear', align_corners=False)

            _preds = sliding_window_inference(
                inputs=volume_tensor,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=_predictor,
                overlap=overlap,
                mode='gaussian'
            )

            _preds = _preds[0, 0, ...]
            _preds = torch.sigmoid(_preds)

            if preds is None:
                preds = _preds
            else:
                preds += _preds

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Average predictions
    preds = preds / len(models)
    
    # Save predictions as .pt file
    pred_path = output_dir / f"{file_path.stem}_pred.pt"
    torch.save(preds.cpu(), pred_path)
    
    # Save metadata as .json file
    meta_path = pred_path.with_suffix('.json')
    metadata = {
        "tomo_id": file_path.stem,
        "z_shape": data.shape[0],
        "y_shape": data.shape[1],
        "x_shape": data.shape[2],
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    console.print(f"\n[bold green]✓[/] Saved inference outputs:")
    console.print(f"  • Predictions: {pred_path}")
    console.print(f"  • Metadata: {meta_path}")
    
    # Post-processing (imported earlier from postprocess lazily)
    console.print("\n[bold blue]Running post-processing...")
    submission = process_predictions(str(output_dir))
    if submission.empty:
        raise RuntimeError("Error during post-processing")
    
    # Convert submission to results format
    results = {
        'coordinates': submission[['tomo_id', 'Motor_Axis_0', 'Motor_Axis_1', 'Motor_Axis_2']],
        'detailed': submission,
        'predictions': preds.cpu().numpy(),
        'masks': (preds.cpu().numpy() > 0.990).astype(np.float32),
        'summary': {
            'confidence': float(submission['max_val'].iloc[0]),
            'threshold': 0.990,
            'coordinates': submission[['Motor_Axis_0', 'Motor_Axis_1', 'Motor_Axis_2']].iloc[0].tolist()
        }
    }
    
    # Save post-processing outputs
    save_outputs(results, output_dir, file_path.stem)
    
    # Generate detailed visualizations if requested
    if args.detailed:  # Use the detailed flag from args
        try:
            from .visualizations import generate_detailed_visualizations_from_inference
            console.print("\n[bold blue]Generating detailed visualizations...")
            
            # Get the model and config from the first ensemble member
            visualization_model = models[0]["model"] if models else None
            model_cfg = models[0]["cfg"] if models else None
            
            # Generate visualizations using file paths and model
            generate_detailed_visualizations_from_inference(
                output_dir=output_dir,
                tomo_id=file_path.stem,
                model=visualization_model,
                model_cfg=model_cfg,
                volume_path=str(file_path),  # original input file
                preds_path=str(output_dir / f"{file_path.stem}_pred.pt")  # prediction file
            )
            console.print("[green]✓[/] Detailed visualizations generated in:")
            console.print(f"  • Visualizations: {output_dir}/visualizations/")
            console.print("  Including:")
            console.print("    - Slice views (top, front, side)")
            console.print("    - Feature maps")
            console.print("    - Encoder-decoder flow visualization")
            
        except Exception as e:
            console.print(f"[yellow]Warning:[/] Failed to generate detailed visualizations: {e}")
            if debug:
                import traceback
                console.print("[red]Detailed error:[/]")
                console.print(traceback.format_exc())
    
    return results

def save_outputs(results: Dict, 
                output_dir: Path,
                prefix: str) -> None:
    """Save all output files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save coordinates CSV
    coords_path = output_dir / f"{prefix}_motor_coords.csv"
    results['coordinates'].to_csv(coords_path, index=False)
    
    # Save detailed results
    detailed_path = output_dir / f"{prefix}_detailed.csv"
    results['detailed'].to_csv(detailed_path, index=False)
    
    # Save prediction maps in both .npy and .mrc formats
    pred_path = output_dir / f"{prefix}_predictions.npy"
    np.save(pred_path, results['predictions'])
    
    pred_mrc_path = output_dir / f"{prefix}_predictions.mrc"
    with mrcfile.new(pred_mrc_path, overwrite=True) as mrc:
        # Ensure data is float32 for MRC format
        mrc.set_data(results['predictions'].astype(np.float32))
    
    # Save masks in both .npy and .mrc formats
    mask_path = output_dir / f"{prefix}_masks.npy"
    np.save(mask_path, results['masks'])
    
    mask_mrc_path = output_dir / f"{prefix}_masks.mrc"
    with mrcfile.new(mask_mrc_path, overwrite=True) as mrc:
        # Ensure data is float32 for MRC format
        mrc.set_data(results['masks'].astype(np.float32))
    
    # Save summary
    summary_path = output_dir / f"{prefix}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results['summary'], f, indent=2)
        
    console.print("\n[bold green]✓[/] Saved output files:")
    console.print(f"  • Coordinates: {coords_path}")
    console.print(f"  • Detailed results: {detailed_path}")
    console.print(f"  • Predictions (NPY): {pred_path}")
    console.print(f"  • Predictions (MRC): {pred_mrc_path}")
    console.print(f"  • Masks (NPY): {mask_path}")
    console.print(f"  • Masks (MRC): {mask_mrc_path}")
    console.print(f"  • Summary: {summary_path}")

def main():
    import argparse
    from rich import print as rprint
    
    parser = argparse.ArgumentParser(
        description="[bold blue]Tomo-Detect:[/] Motor coordinate detection in tomography data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input",
        nargs="+",
        help="One or more input file paths (.npy, .mrc) or zip files containing multiple inputs"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory path",
        default="./tomo_detect_output"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--preserve-range",
        dest='preserve_range',
        action='store_true',
        help=(
            "Preserve the original numeric range of the input (do not normalize to [0,1]).\n"
            "This matches the standalone_inference behaviour where the model divides by 255 internally."
        ),
    )
    parser.add_argument(
        "--no-preserve-range",
        dest='preserve_range',
        action='store_false',
        help="Normalize input values to [0,1] before inference (opposite of --preserve-range).",
    )
    parser.set_defaults(preserve_range=True)
    
    parser.add_argument(
        "--device",
        choices=['cuda', 'cpu'],
        help="Device to use for inference. Default: Use CUDA if available"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (faster but less accurate)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed visualizations including feature maps, encoder-decoder flow, and tail analysis"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    try:
        # Print system info
        print_system_info()
        
        # Set device
        if args.device:
            device = args.device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Process all input paths
        for input_file in args.input:
            try:
                input_path = Path(input_file)
                if not input_path.exists():
                    console.print(f"[yellow]Warning:[/] Input path does not exist: {input_path}")
                    continue

                console.print(f"\n[bold blue]Processing input:[/] {input_path}")
                
                # Create output directory for this input
                input_output_dir = Path(args.output) / input_path.stem
                input_output_dir.mkdir(parents=True, exist_ok=True)

                if input_path.is_file():
                    if input_path.suffix.lower() == '.zip':
                        # Handle zip file
                        with zipfile.ZipFile(input_path) as zf:
                            for fname in zf.namelist():
                                if fname.lower().endswith(('.npy', '.mrc')):
                                    with zf.open(fname) as f:
                                        temp_path = input_output_dir / 'temp' / fname
                                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                                        with open(temp_path, 'wb') as tf:
                                            tf.write(f.read())
                                        process_input_file(
                                            temp_path,
                                            input_output_dir,
                                            batch_size=args.batch_size,
                                            device=device,
                                            debug=args.debug,
                                            preserve_range=args.preserve_range
                                        )
                    elif input_path.suffix.lower() in ['.npy', '.mrc']:
                        # Process single file
                        process_input_file(
                            input_path,
                            input_output_dir,
                            batch_size=args.batch_size,
                            device=device,
                            debug=args.debug,
                            test=args.test,
                            preserve_range=args.preserve_range,
                            args=args
                        )
                    else:
                        console.print(f"[yellow]Warning:[/] Unsupported file type: {input_path}")
                        continue
                
                elif input_path.is_dir():
                    # Check for image files in directory
                    image_files = []
                    for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                        image_files.extend(input_path.glob(f'*{ext}'))
                        image_files.extend(input_path.glob(f'*{ext.upper()}'))
                    
                    if image_files:
                        console.print(f"Found {len(image_files)} image files in directory")
                        try:
                            process_input_file(
                                file_path,
                                input_output_dir / file_path.stem,
                                batch_size=args.batch_size,
                                device=device,
                                debug=args.debug,
                                preserve_range=args.preserve_range,
                                args=args
                            )
                        except Exception as e:
                            console.print(f"[bold red]Error processing image stack:[/] {str(e)}")
                    else:
                        # Look for .npy and .mrc files
                        found_files = False
                        for file_path in input_path.glob('*'):
                            if file_path.suffix.lower() in ['.npy', '.mrc']:
                                found_files = True
                                process_input_file(
                                    file_path,
                                    input_output_dir / file_path.stem,
                                    batch_size=args.batch_size,
                                    device=device,
                                    debug=args.debug,
                                    preserve_range=args.preserve_range
                                )
                        if not found_files:
                            console.print(f"[yellow]Warning:[/] No supported files found in directory: {input_path}")

            except Exception as e:
                console.print(f"[bold red]Error processing {input_path}:[/] {str(e)}")
                if args.debug:
                    logging.exception("Detailed error:")
        
        console.print("\n[bold green]✓[/] Processing completed successfully!")
        
    except Exception as e:
        if args.debug:
            logging.exception("An error occurred:")
        else:
            console.print(f"\n[bold red]Error:[/] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()