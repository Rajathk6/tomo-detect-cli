#!/usr/bin/env python3
"""
Helper script to load a model checkpoint, run a single forward pass on a tomogram
and generate the encoder/decoder feature visualizations.

Usage (PowerShell):
    python .\scripts\run_visualizations.py --tomo aba2013-04-06-9 --model tomo_detect\models\r3d200_704_350984_epoch400.pt

If --model is omitted the script picks the first .pt under `tomo_detect/models/`.
"""
import sys
from pathlib import Path
import argparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tomo_detect.visualizations import (
    create_feature_visualizations,
    FeatureExtractor,
    create_encoder_decoder_flow,
    create_detailed_feature_analysis,
    create_synthesized_intermediates,
)
from tomo_detect.inference import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tomo", required=True, help="tomo id folder under tomo_detect_output, e.g. aba2013-04-06-9")
    p.add_argument("--model", default=None, help="path to .pt checkpoint (optional)")
    p.add_argument("--output", default=str(ROOT / "tomo_detect_output"), help="root output folder")
    p.add_argument("--synth", type=int, default=3, help="number of interpolated steps to synthesize between start and end features")
    p.add_argument("--model-index", type=int, default=None, help="If model dir contains multiple checkpoints, select one by index (0-based)")
    p.add_argument("--all-models", action='store_true', help="If set, process all .pt files found in the models folder and write per-model subfolders")
    args = p.parse_args()

    output_root = Path(args.output)
    tomo_dir = output_root / args.tomo
    if not tomo_dir.exists():
        raise FileNotFoundError(f"Tomo output folder not found: {tomo_dir}")

    # find input and prediction files
    raw_npy = tomo_dir / f"{args.tomo}_raw.npy"
    pred_pt = tomo_dir / f"{args.tomo}_pred.pt"
    if not raw_npy.exists():
        raise FileNotFoundError(f"Raw input not found: {raw_npy}")
    if not pred_pt.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_pt}")

    # choose model
    model_path = None
    if args.model:
        model_path = Path(args.model)
    else:
        models_dir = ROOT / "tomo_detect" / "models"
        pts = sorted([p for p in models_dir.iterdir() if p.suffix == ".pt"])
        if not pts:
            raise FileNotFoundError(f"No .pt files found in {models_dir}")
        model_path = pts[0]

    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine model paths to process. If --all-models, iterate a models directory; otherwise one model.
    model_paths = []
    if args.all_models:
        # If user passed --model pointing to a directory, use that; otherwise default models folder
        if args.model and Path(args.model).is_dir():
            models_dir = Path(args.model)
        else:
            models_dir = ROOT / "tomo_detect" / "models"
        pts = sorted([p for p in models_dir.iterdir() if p.suffix == ".pt"])
        if not pts:
            raise FileNotFoundError(f"No .pt files found in {models_dir}")
        model_paths = pts
    else:
        model_paths = [model_path]

    # If user requested a specific index, pick that one
    if args.model_index is not None and len(model_paths) > 1:
        idx = int(args.model_index)
        if idx < 0 or idx >= len(model_paths):
            raise IndexError(f"--model-index {idx} out of range (0..{len(model_paths)-1})")
        model_paths = [model_paths[idx]]

    # Load input and predictions once (shared across model runs)
    raw = np.load(str(raw_npy))
    preds = torch.load(str(pred_pt), weights_only=False)

    # Create basic visualizations (slices, surface, etc.) without model
    try:
        create_feature_visualizations(raw, preds, tomo_dir, model=None)
    except Exception as e:
        print("Basic visualization generation failed:", e)

    # For each model checkpoint, load and create model-specific visualizations
    for mp in model_paths:
        print(f"Loading model from: {mp} on device {device}")
        try:
            model, cfg = load_model(str(mp), device)
        except Exception as e:
            print(f"Failed to load model {mp}:", e)
            continue

        # prepare model input from raw (assume raw is Z,Y,X)
        model_input = torch.from_numpy(raw.astype('float32')).unsqueeze(0).unsqueeze(0).to(device)
        model = model.to(device)

        try:
            feature_extractor = FeatureExtractor(model)
            with torch.no_grad():
                _ = model(model_input)

            # choose visualization output folder
            if args.all_models:
                vis_out = tomo_dir / 'visualizations' / mp.stem
            else:
                vis_out = tomo_dir / 'visualizations'

            vis_out.mkdir(parents=True, exist_ok=True)

            try:
                create_encoder_decoder_flow(feature_extractor, raw, preds, vis_out / 'encoder_decoder_flow.png')
            except Exception as e:
                print('create_encoder_decoder_flow failed:', e)

            try:
                create_detailed_feature_analysis(feature_extractor, vis_out)
            except Exception as e:
                print('create_detailed_feature_analysis failed:', e)

            # Optional: synthesize interpolated intermediate stages to show progress
            synth_steps = int(args.synth) if hasattr(args, 'synth') else 3
            try:
                create_synthesized_intermediates(feature_extractor, vis_out, n_steps=synth_steps)
            except Exception as e:
                print('create_synthesized_intermediates failed:', e)

            print(f"Visualizations written to: {vis_out}")

        except Exception as e:
            print("Model-based visualization generation failed for", mp, e)


if __name__ == '__main__':
    main()
