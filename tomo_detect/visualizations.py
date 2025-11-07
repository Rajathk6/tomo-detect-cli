import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import mrcfile
from pathlib import Path
from typing import Union
import logging

# Disable matplotlib and PIL debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Configure matplotlib to use a non-interactive backend
plt.switch_backend('agg')

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.encoder_features = []  # Will store (name, feature) tuples for encoder
        self.decoder_features = []  # Will store (name, feature) tuples for decoder
        self._hooks = []
        
        # Register detailed encoder hooks to capture transformations
        def make_encoder_hooks(stage_num, block):
            def input_hook(module, inputs, outputs):
                self.encoder_features.append(
                    (f"encoder_{stage_num}_input", inputs[0].detach().cpu())
                )
            
            def conv1_hook(module, inputs, outputs):
                self.encoder_features.append(
                    (f"encoder_{stage_num}_conv1", outputs.detach().cpu())
                )
            
            def bn1_hook(module, inputs, outputs):
                self.encoder_features.append(
                    (f"encoder_{stage_num}_bn1", outputs.detach().cpu())
                )
            
            def conv2_hook(module, inputs, outputs):
                self.encoder_features.append(
                    (f"encoder_{stage_num}_conv2", outputs.detach().cpu())
                )
            
            def conv3_hook(module, inputs, outputs):
                self.encoder_features.append(
                    (f"encoder_{stage_num}_conv3", outputs.detach().cpu())
                )
            
            def final_hook(module, inputs, outputs):
                self.encoder_features.append(
                    (f"encoder_{stage_num}_final", outputs.detach().cpu())
                )
            
            # Register hooks for each transformation and filter out None values
            hooks = [
                block.register_forward_hook(input_hook),
                block.conv1.register_forward_hook(conv1_hook) if hasattr(block, 'conv1') else None,
                block.bn1.register_forward_hook(bn1_hook) if hasattr(block, 'bn1') else None,
                block.conv2.register_forward_hook(conv2_hook) if hasattr(block, 'conv2') else None,
                block.conv3.register_forward_hook(conv3_hook) if hasattr(block, 'conv3') else None,
                block.register_forward_hook(final_hook)
            ]
            self._hooks.extend([h for h in hooks if h is not None])
        
        # Assign sequential encoder stage numbers and register hooks
        encoder_stage_counter = 1

        # Initial conv layer (stage 1)
        if hasattr(model.backbone, 'conv1'):
            make_encoder_hooks(encoder_stage_counter, model.backbone.conv1)
            encoder_stage_counter += 1

        # Register hooks for each ResNet block with sequential numbering
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(model.backbone, layer_name):
                layer = getattr(model.backbone, layer_name)
                for block in layer:
                    make_encoder_hooks(encoder_stage_counter, block)
                    encoder_stage_counter += 1
        
                # Enhanced decoder hooks with stage names
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'blocks'):
            for idx, block in enumerate(model.decoder.blocks):
                stage_num = idx + 1
                
                # Capture input before upsampling
                def make_pre_forward_hook(stage):
                    def hook(module, inputs, outputs):
                        # inputs[0] is the input before upsampling
                        self.decoder_features.append(
                            (f"stage_{stage}_input", inputs[0].detach().cpu())
                        )
                    return hook
                
                # Capture upsampled features
                def make_up_forward_hook(stage):
                    def hook(module, inputs, outputs):
                        # outputs is the upsampled features
                        self.decoder_features.append(
                            (f"stage_{stage}_upsampled", outputs.detach().cpu())
                        )
                    return hook
                
                # Capture post-skip-connection features
                def make_skip_forward_hook(stage):
                    def hook(module, inputs, outputs):
                        # inputs[0] contains concatenated skip features
                        self.decoder_features.append(
                            (f"stage_{stage}_skipcat", inputs[0].detach().cpu())
                        )
                    return hook
                
                # Capture intermediate conv features
                def make_conv1_hook(stage):
                    def hook(module, inputs, outputs):
                        self.decoder_features.append(
                            (f"stage_{stage}_conv1", outputs.detach().cpu())
                        )
                    return hook
                
                # Capture final features
                def make_final_hook(stage):
                    def hook(module, inputs, outputs):
                        self.decoder_features.append(
                            (f"stage_{stage}_final", outputs.detach().cpu())
                        )
                    return hook
                
                # Register all hooks for this decoder stage
                if hasattr(block, 'upsample'):
                    # Input before upsampling
                    hook = block.register_forward_hook(make_pre_forward_hook(stage_num))
                    self._hooks.append(hook)
                    
                    # After upsampling
                    hook = block.upsample.register_forward_hook(make_up_forward_hook(stage_num))
                    self._hooks.append(hook)
                
                # After skip connection concatenation
                if hasattr(block, 'conv1'):
                    hook = block.conv1.register_forward_hook(make_skip_forward_hook(stage_num))
                    self._hooks.append(hook)
                    
                    # After first convolution
                    hook = block.conv1.register_forward_hook(make_conv1_hook(stage_num))
                    self._hooks.append(hook)
                
                # Final output
                if hasattr(block, 'conv2'):
                    hook = block.conv2.register_forward_hook(make_final_hook(stage_num))
                    self._hooks.append(hook)
    
    def get_feature_by_name(self, name):
        """Helper to get a feature by its name"""
        for n, feat in self.encoder_features + self.decoder_features:
            if n == name:
                return feat
        return None
    
    def get_stage_features(self, stage_num, stage_type='final'):
        """Helper to get decoder features for a specific stage"""
        name = f"stage_{stage_num}_{stage_type}"
        return self.get_feature_by_name(name)
    
    def __del__(self):
        # Clean up hooks when the object is destroyed
        try:
            for hook in self._hooks:
                if hook is not None:
                    hook.remove()
        except Exception:
            pass  # Ignore cleanup errors during deletion

def create_feature_visualizations(input_data, predictions, output_dir, model=None):
    """Create comprehensive feature visualizations if model is provided."""
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle input data format
    if isinstance(input_data, np.ndarray):
        if input_data.ndim == 3:  # DHW
            model_input = torch.from_numpy(input_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # Add B and C dimensions
            vis_input = input_data.astype(np.float32)  # Keep original 3D for visualization
        elif input_data.ndim == 4:  # CDHW or BDHW
            model_input = torch.from_numpy(input_data.astype(np.float32)).unsqueeze(0)  # Add batch dimension
            vis_input = input_data.squeeze().astype(np.float32)  # Convert to 3D for visualization
        elif input_data.ndim == 5:  # BCDHW
            model_input = torch.from_numpy(input_data.astype(np.float32))
            vis_input = input_data.squeeze().astype(np.float32)  # Convert to 3D for visualization
    elif isinstance(input_data, torch.Tensor):
        if input_data.ndim == 3:  # DHW
            model_input = input_data.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add B and C dimensions
            vis_input = input_data.detach().cpu().numpy().astype(np.float32)  # Keep original 3D for visualization
        elif input_data.ndim == 4:  # CDHW or BDHW
            model_input = input_data.to(dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            vis_input = input_data.squeeze().detach().cpu().numpy().astype(np.float32)  # Convert to 3D for visualization
        elif input_data.ndim == 5:  # BCDHW
            model_input = input_data.to(dtype=torch.float32)
            vis_input = input_data.squeeze().detach().cpu().numpy().astype(np.float32)  # Convert to 3D for visualization
    else:
        raise ValueError(f"Unsupported input_data type: {type(input_data)}")
    
    # Handle predictions format
    if isinstance(predictions, torch.Tensor):
        vis_pred = predictions.detach().cpu().numpy().astype(np.float32)
    elif isinstance(predictions, np.ndarray):
        vis_pred = predictions.astype(np.float32)
    else:
        raise ValueError(f"Unsupported predictions type: {type(predictions)}")
    
    # Ensure predictions are 3D for visualization
    if vis_pred.ndim > 3:
        vis_pred = vis_pred.squeeze()  # Remove any extra dimensions
    if vis_pred.ndim != 3:
        raise ValueError(f"Predictions must be 3D after squeezing. Got shape: {vis_pred.shape}")
    
    # Basic visualizations (always created)
    create_slice_visualization(vis_input, vis_dir / 'slices.png')
    # 3D surface and intensity profile visualizations removed per user request
    
    # Model-specific visualizations
    if model is not None:
        feature_extractor = FeatureExtractor(model)
        with torch.no_grad():
            if torch.cuda.is_available():
                model_input = model_input.cuda()
                model = model.cuda()
            _ = model(model_input)
        
        create_encoder_decoder_flow(feature_extractor, vis_input, vis_pred, 
                                  vis_dir / 'encoder_decoder_flow.png')
        create_detailed_feature_analysis(feature_extractor, vis_dir)

def create_slice_visualization(data, save_path):
    """Create a multi-slice visualization showing different views."""
    try:
        # Convert to numpy and ensure float32
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy().astype(np.float32)
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        elif data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Ensure data is 3D
        if data.ndim > 3:
            data = data.squeeze()
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D after squeezing. Got shape: {data.shape}")
        
        # Get center slices
        z_slice = data[data.shape[0]//2, :, :]
        y_slice = data[:, data.shape[1]//2, :]
        x_slice = data[:, :, data.shape[2]//2]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Tomogram Multi-View Visualization', fontsize=16)
        
        # Normalize slices for better visualization
        def normalize_slice(slice_data):
            if not isinstance(slice_data, np.ndarray):
                slice_data = np.array(slice_data, dtype=np.float32)
            elif slice_data.dtype != np.float32:
                slice_data = slice_data.astype(np.float32)
            slice_min = slice_data.min()
            slice_max = slice_data.max()
            if slice_max > slice_min:
                return (slice_data - slice_min) / (slice_max - slice_min)
            return slice_data
        
        axes[0, 0].imshow(normalize_slice(z_slice), cmap='gray')
        axes[0, 0].set_title('Top View (Z-slice)')
        
        axes[0, 1].imshow(normalize_slice(y_slice), cmap='gray')
        axes[0, 1].set_title('Front View (Y-slice)')
        
        axes[1, 0].imshow(normalize_slice(x_slice), cmap='gray')
        axes[1, 0].set_title('Side View (X-slice)')
        
        # Plot histogram of non-zero values for better visualization
        non_zero_data = data[data != 0].ravel()
        if len(non_zero_data) > 0:
            axes[1, 1].hist(non_zero_data.astype(np.float32), bins=50, color='blue', alpha=0.7)
        else:
            axes[1, 1].hist(data.ravel().astype(np.float32), bins=50, color='blue', alpha=0.7)
        axes[1, 1].set_title('Intensity Distribution')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise RuntimeError(f"Failed to create slice visualization: {str(e)}") from e

def create_3d_surface_plot(*args, **kwargs):
    """Removed: 3D surface plotting disabled by user request.

    This stub remains to preserve API compatibility but does nothing.
    """
    return

def create_intensity_profile(*args, **kwargs):
    """Removed: intensity profile plotting disabled by user request.

    This stub remains for API compatibility but does nothing.
    """
    return

def create_contour_visualization(data, save_path):
    """Create a multi-level contour visualization."""
    # Contour visualization intentionally removed to reduce clutter — detailed encoder/decoder visuals provide
    # richer insight. Kept the function stub for backwards compatibility but it does nothing now.
    return

def create_encoder_decoder_flow(feature_extractor, input_data, output_data, save_path):
    """Create a visualization of the encoder-decoder feature transformation flow."""
    try:
        # Convert input/output data to numpy and ensure float32
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy().astype(np.float32)
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
        elif input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
            
        # Group encoder features by stage
        encoder_stages = {}
        for name, feat in feature_extractor.encoder_features:
            if name.startswith('encoder_'):
                parts = name.split('_')
                stage_num = int(parts[1])
                stage_type = '_'.join(parts[2:])
                
                if stage_num not in encoder_stages:
                    encoder_stages[stage_num] = {}
                encoder_stages[stage_num][stage_type] = feat
        
        # Sort stages and select intervals
        selected_encoder_stages = []
        for stage_num in sorted(encoder_stages.keys()):
            if stage_num == 1 or stage_num % 100 == 0:
                selected_encoder_stages.append((stage_num, encoder_stages[stage_num]))
        
        # Calculate layout dimensions
        num_encoder_rows = len(selected_encoder_stages)  # One row per encoder stage
        
        # Get decoder stage numbers dynamically from captured features
        decoder_stage_nums = sorted({
            int(name.split('_')[1]) for name, _ in feature_extractor.decoder_features if name.startswith('stage_')
        })
        # Fallback to stages 1-4 if none found
        if not decoder_stage_nums:
            decoder_stage_nums = list(range(1, 5))

        decoder_stages = []
        for stage_num in decoder_stage_nums:
            stage_features = {
                stage_type: next((feat for name, feat in feature_extractor.decoder_features 
                                if name == f"stage_{stage_num}_{stage_type}"), None)
                for stage_type in ['input', 'upsampled', 'skipcat', 'conv1', 'final']
            }
            decoder_stages.append((stage_num, stage_features))
        
        # Define transformation types
        encoder_transform_types = ['input', 'conv1', 'bn1', 'conv2', 'conv3', 'final']
        decoder_transform_types = ['input', 'upsampled', 'skipcat', 'conv1', 'final']
        
        # Calculate proper figure size and layout
        num_total_rows = num_encoder_rows + len(decoder_stages)
        num_cols = max(len(encoder_transform_types), len(decoder_transform_types))
        fig = plt.figure(figsize=(4 * num_cols, 4 * num_total_rows))  # Adjust figure size based on content
        gs = GridSpec(num_total_rows, num_cols, figure=fig, hspace=0.4)
        
        # Plot encoder stages: each encoder selected stage gets a full row of transformations
        for stage_idx, (stage_num, stage_features) in enumerate(selected_encoder_stages):
            row = stage_idx
            # Add stage label on left side
            fig.text(0.01, 1 - (row + 0.5) / num_total_rows, f'Encoder Stage {stage_num}', 
                    ha='left', va='center', fontsize=12)

            for col_idx, feat_type in enumerate(encoder_transform_types):
                ax = fig.add_subplot(gs[row, col_idx])
                feat = stage_features.get(feat_type)
                if feat is None:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f'{feat_type}\n-')
                    continue

                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy()

                # ensure shape (B,C,Z,Y,X)
                if feat.ndim == 4:
                    # (C,Z,Y,X) -> add batch
                    feat = np.expand_dims(feat, 0)

                # Protect against small channel dims
                try:
                    feat_map = feat[0, 0]
                except Exception:
                    ax.text(0.5, 0.5, 'bad-shape', ha='center', va='center')
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f'{feat_type}\n?')
                    continue

                z_mid = feat_map.shape[0] // 2
                im = ax.imshow(feat_map[z_mid], cmap='viridis')
                ch_count = feat.shape[1] if feat.ndim >= 2 else 1
                ax.set_title(f'{feat_type}\n{ch_count} channels')
                plt.colorbar(im, ax=ax)
                ax.axis('off')

        # Plot decoder stages: each decoder stage gets its full row; align to same number of columns
        for stage_idx, (stage_num, stage_features) in enumerate(decoder_stages):
            row = num_encoder_rows + stage_idx
            fig.text(0.01, 1 - (row + 0.5) / num_total_rows, f'Decoder Stage {stage_num}', 
                    ha='left', va='center', fontsize=12)

            for col_idx in range(num_cols):
                ax = fig.add_subplot(gs[row, col_idx])
                # map decoder transform to a column: use decoder_transform_types mapped to cols 0..len-1
                if col_idx < len(decoder_transform_types):
                    feat_type = decoder_transform_types[col_idx]
                else:
                    feat_type = None

                if feat_type is None or feat_type not in stage_features:
                    # leave empty or put placeholder
                    ax.text(0.5, 0.5, '', ha='center', va='center')
                    ax.set_xticks([]); ax.set_yticks([])
                    continue

                feat = stage_features.get(feat_type)
                if feat is None:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f'{feat_type}\n-')
                    continue

                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy()

                if feat.ndim == 4:
                    feat = np.expand_dims(feat, 0)

                try:
                    feat_map = feat[0, 0]
                except Exception:
                    ax.text(0.5, 0.5, 'bad-shape', ha='center', va='center')
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f'{feat_type}\n?')
                    continue

                z_mid = feat_map.shape[0] // 2
                im = ax.imshow(feat_map[z_mid], cmap='viridis')
                ch_count = feat.shape[1] if feat.ndim >= 2 else 1
                ax.set_title(f'{feat_type}\n{ch_count} channels')
                plt.colorbar(im, ax=ax)
                ax.axis('off')
        
        plt.suptitle('Encoder-Decoder Feature Transformations', fontsize=16, y=0.98)
        # Save with extra padding to accommodate labels
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise RuntimeError(f"Failed to create encoder-decoder flow visualization: {str(e)}") from e


def _select_stage_samples(all_nums, max_samples: int = 8):
    """Select up to max_samples stage numbers evenly across the available stages.

    all_nums: sorted list of integers (stage numbers).
    Returns a list of selected stage numbers (sorted).
    """
    if not all_nums:
        return []
    if len(all_nums) <= max_samples:
        return list(all_nums)
    idxs = np.linspace(0, len(all_nums) - 1, num=max_samples)
    idxs = np.unique(np.round(idxs).astype(int))
    selected = [all_nums[i] for i in idxs]
    # Ensure first and last are included
    if selected[0] != all_nums[0]:
        selected[0] = all_nums[0]
    if selected[-1] != all_nums[-1]:
        selected[-1] = all_nums[-1]
    # Remove duplicates and sort
    selected = sorted(list(dict.fromkeys(selected)))
    return selected

def create_detailed_feature_analysis(feature_extractor, output_dir):
    """Create detailed visualizations of features at each stage."""
    # Save detailed per-stage images into the provided output_dir (do NOT create a separate feature_maps folder)
    feature_dir = Path(output_dir)
    # Group encoder features by stage
    encoder_stages = {}
    for name, feat in feature_extractor.encoder_features:
        if name.startswith('encoder_'):
            parts = name.split('_')
            stage_num = int(parts[1])
            stage_type = '_'.join(parts[2:])
            if stage_num not in encoder_stages:
                encoder_stages[stage_num] = {}
            encoder_stages[stage_num][stage_type] = feat

    # Choose a small set of evenly distributed encoder stages to visualize (includes first and final)
    all_encoder_nums = sorted(encoder_stages.keys())
    selected_encoder_nums = _select_stage_samples(all_encoder_nums, max_samples=8)

    # Create detailed visualizations for selected encoder stages (detailed panels only; no simple overview files)
    for stage_num in selected_encoder_nums:
        # Detailed transformations only
        visualize_encoder_stage(feature_extractor, stage_num, feature_dir)
    
    # Group decoder features by stage
    decoder_stages = {}
    for stage_data in feature_extractor.decoder_features:
        stage_name, features = stage_data
        stage_num = int(stage_name.split('_')[1])
        stage_type = '_'.join(stage_name.split('_')[2:])
        
        if stage_num not in decoder_stages:
            decoder_stages[stage_num] = {}
        decoder_stages[stage_num][stage_type] = features
    
    # Create detailed visualizations for each decoder stage
    for stage_num, stage_features in sorted(decoder_stages.items()):
        # Basic feature map visualization (saved into output_dir)
        filename = f"decoder_stage_{stage_num}.png"
        save_path = feature_dir / filename
        title = f"Decoder Stage {stage_num}"

        # Show final features in overview
        if 'final' in stage_features:
            visualize_feature_maps(stage_features['final'], title, save_path)

        # Create detailed stage analysis showing all transformations
        visualize_decoder_stage(feature_extractor, stage_num, feature_dir)

def visualize_feature_maps(features, title, save_path, total_stages=None):
    """Create a grid visualization of feature maps."""
    try:
        # Handle (name, features) tuple format
        if isinstance(features, tuple) and len(features) == 2:
            _, features = features
        
        # Convert to numpy and ensure float32
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy().astype(np.float32)
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        elif features.dtype != np.float32:
            features = features.astype(np.float32)
        
        # Sample features at intervals if there are too many
        total_channels = features.shape[1]
        if total_channels > 16:
            step = max(1, total_channels // 16)  # Sample max 16 channels
            channel_indices = list(range(0, total_channels, step))
            if total_channels - 1 not in channel_indices:
                channel_indices.append(total_channels - 1)  # Always include last channel
            features = features[:, channel_indices]
        
        n_features = min(16, features.shape[1])
        size = int(np.ceil(np.sqrt(n_features)))
        
        fig, axes = plt.subplots(size, size, figsize=(15, 15))
        title_text = f'{title}\n(Sampled {n_features} channels out of {total_channels})'
        if total_stages:
            title_text += f'\n(Stage {title.split()[-1]} of {total_stages} total stages)'
        fig.suptitle(title_text, fontsize=16)
        
        # Make axes array 2D if it isn't already
        if size == 1:
            axes = np.array([[axes]])
        elif len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(n_features):
            i, j = divmod(idx, size)
            feature = features[0, idx]
            if not isinstance(feature, np.ndarray) or feature.dtype != np.float32:
                feature = np.array(feature, dtype=np.float32)
            axes[i, j].imshow(feature[feature.shape[0]//2], cmap='viridis')
            axes[i, j].axis('off')
        
        # Remove empty subplots
        for idx in range(n_features, size * size):
            i, j = divmod(idx, size)
            axes[i, j].remove()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise RuntimeError(f"Failed to visualize feature maps: {str(e)}") from e

def visualize_decoder_stage(feature_extractor, stage_num, output_dir):
    """
    Enhanced visualization for decoder stages showing all intermediate transformations:
    - Input features before upsampling
    - Upsampled features
    - Skip connection concatenation
    - Conv1 features
    - Final output features
    """
    try:
        # Get all features for this stage in sequence
        stages = [
            ('input', 'Input Features'),
            ('upsampled', 'After Upsampling'),
            ('skipcat', 'Skip Connection'),
            ('conv1', 'After Conv1'),
            ('final', 'Final Output')
        ]
        
        # Create figure with five subplots for each transformation
        fig = plt.figure(figsize=(20, 8))
        plt.suptitle(f'Decoder Stage {stage_num} Feature Transformations', fontsize=14)
        
        # Plot each stage
        for idx, (stage_type, title) in enumerate(stages, 1):
            stage_name = f'stage_{stage_num}_{stage_type}'
            feat = feature_extractor.get_feature_by_name(stage_name)
            
            if feat is not None:
                ax = plt.subplot(1, 5, idx)
                
                # Convert to numpy if needed
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy()
                
                # Get middle slice for visualization
                z_mid = feat.shape[2] // 2
                feature_map = feat[0, 0, z_mid]  # Take first channel for visualization
                
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'{title}\n{feat.shape[1]} channels')
                plt.colorbar(im, ax=ax)
                ax.axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'decoder_stage_{stage_num}_detailed.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise RuntimeError(f"Failed to create decoder stage visualization: {str(e)}") from e

def visualize_encoder_stage(feature_extractor, stage_num, output_dir):
    """
    Visualization for encoder stages showing intermediate transformations:
    - Input
    - Conv1
    - BN1
    - Conv2
    - Conv3
    - Final
    """
    try:
        stages = [
            ('input', 'Input Features'),
            ('conv1', 'After Conv1'),
            ('bn1', 'After BN1'),
            ('conv2', 'After Conv2'),
            ('conv3', 'After Conv3'),
            ('final', 'Final Output')
        ]

        fig = plt.figure(figsize=(24, 6))
        plt.suptitle(f'Encoder Stage {stage_num} Feature Transformations', fontsize=14)

        for idx, (stage_type, title) in enumerate(stages, 1):
            name = f'encoder_{stage_num}_{stage_type}'
            feat = feature_extractor.get_feature_by_name(name)
            if feat is not None:
                ax = plt.subplot(1, 6, idx)
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy()
                z_mid = feat.shape[2] // 2
                feature_map = feat[0, 0, z_mid]
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'{title}\n{feat.shape[1]} channels')
                plt.colorbar(im, ax=ax)
                ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'encoder_stage_{stage_num}_detailed.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        plt.close()
        raise RuntimeError(f"Failed to create encoder stage visualization: {str(e)}") from e


def _to_tensor(feat):
    """Convert feature (numpy or torch) to torch.FloatTensor on CPU."""
    if isinstance(feat, torch.Tensor):
        t = feat.detach().cpu().float()
    else:
        t = torch.from_numpy(np.array(feat)).float()
    return t


def _match_and_interpolate(a: torch.Tensor, b: torch.Tensor):
    """Match channel and spatial shapes of two feature tensors by truncation/padding and trilinear interpolate spatially.

    Both a and b expected shape (B, C, Z, Y, X).
    Returns (a_resized, b_resized) as torch tensors on CPU.
    """
    # ensure 5D
    if a.ndim != 5 or b.ndim != 5:
        raise ValueError("Features must be 5D tensors (B,C,Z,Y,X)")

    # match channels by truncation or padding
    ca = a.shape[1]
    cb = b.shape[1]
    cmin = min(ca, cb)
    if ca != cmin:
        a = a[:, :cmin]
    if cb != cmin:
        b = b[:, :cmin]

    # match spatial dims: pick max dims and interpolate smaller to larger
    za, ya, xa = a.shape[2:]
    zb, yb, xb = b.shape[2:]
    zt = max(za, zb)
    yt = max(ya, yb)
    xt = max(xa, xb)

    if (za, ya, xa) != (zt, yt, xt):
        a = torch.nn.functional.interpolate(a, size=(zt, yt, xt), mode='trilinear', align_corners=False)
    if (zb, yb, xb) != (zt, yt, xt):
        b = torch.nn.functional.interpolate(b, size=(zt, yt, xt), mode='trilinear', align_corners=False)

    return a, b


def synthesize_between_and_save(feat_a, feat_b, n_steps, save_prefix, output_dir):
    """Create n_steps interpolated feature tensors between feat_a and feat_b and save images.

    feat_a / feat_b: numpy or torch tensors (B,C,Z,Y,X) or shapes convertible.
    """
    try:
        a = _to_tensor(feat_a)
        b = _to_tensor(feat_b)
        # match shapes
        a, b = _match_and_interpolate(a, b)

        for k in range(1, n_steps + 1):
            alpha = float(k) / (n_steps + 1)
            interp = (1.0 - alpha) * a + alpha * b
            # save visualization for this interpolated tensor
            save_name = f"{save_prefix}_interp_{k}.png"
            save_path = Path(output_dir) / save_name
            # visualize_feature_maps expects numpy/torch in same format
            visualize_feature_maps(interp, f"{save_prefix} interp {k}", save_path)
    except Exception as e:
        raise RuntimeError(f"Failed to synthesize intermediate features: {str(e)}") from e


def create_synthesized_intermediates(feature_extractor, output_dir, n_steps: int = 3):
    """Create synthesized intermediate visualizations for encoder and decoder stages.

    For encoder: synthesize between 'input' and 'final' of selected encoder stages.
    For decoder: synthesize between 'input' and 'final' (or 'upsampled' and 'final' if available).
    """
    # Save synthesized intermediates into the provided output_dir (do NOT create a separate feature_maps folder)
    feature_dir = Path(output_dir)

    # Encoder: collect grouped stages
    enc = {}
    for name, feat in feature_extractor.encoder_features:
        if name.startswith('encoder_'):
            parts = name.split('_')
            sn = int(parts[1])
            st = '_'.join(parts[2:])
            enc.setdefault(sn, {})[st] = feat

    # select same stages as create_detailed_feature_analysis (evenly spaced)
    enc_nums = sorted(enc.keys())
    selected = _select_stage_samples(enc_nums, max_samples=8)

    for sn in selected:
        stage_feats = enc.get(sn, {})
        # choose endpoints: prefer 'input' and 'final'
        a = stage_feats.get('input')
        b = stage_feats.get('final')
        if a is not None and b is not None:
            synth_prefix = f"encoder_stage_{sn}"
            synth_out = feature_dir
            synthesize_between_and_save(a, b, n_steps, synth_prefix, synth_out)

    # Decoder: group by stage
    dec = {}
    for name, feat in feature_extractor.decoder_features:
        if name.startswith('stage_'):
            parts = name.split('_')
            sn = int(parts[1])
            st = '_'.join(parts[2:])
            dec.setdefault(sn, {})[st] = feat

    for sn, stage_feats in dec.items():
        # pick upsampled vs final if possible, else input vs final
        a = stage_feats.get('upsampled', stage_feats.get('input'))
        b = stage_feats.get('final')
        if a is not None and b is not None:
            synth_prefix = f"decoder_stage_{sn}"
            synth_out = feature_dir
            synthesize_between_and_save(a, b, n_steps, synth_prefix, synth_out)

def generate_detailed_visualizations_from_inference(output_dir: Union[str, Path], 
                                                 tomo_id: str,
                                                 model=None,
                                                 model_cfg=None,
                                                 volume_path=None,
                                                 preds_path=None,
                                                 model_index: int = None,
                                                 iterate_ensemble: bool = False):
    """Generate detailed visualizations for inference results."""
    output_dir = Path(output_dir)
    
    try:
        # Load input data
        if volume_path:
            if str(volume_path).endswith('.npy'):
                input_data = np.load(volume_path)
            elif str(volume_path).endswith('.mrc'):
                with mrcfile.open(volume_path) as mrc:
                    input_data = mrc.data
            else:
                raise ValueError(f"Unsupported input file format: {volume_path}")
        else:
            # Try to find input data in output directory
            npy_path = output_dir / f"{tomo_id}_raw.npy"
            mrc_path = output_dir / f"{tomo_id}_raw.mrc"
            if npy_path.exists():
                input_data = np.load(npy_path)
            elif mrc_path.exists():
                with mrcfile.open(mrc_path) as mrc:
                    input_data = mrc.data
            else:
                raise FileNotFoundError("Could not find input data file")
                
        # Convert to numpy array and ensure float32
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
        elif input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # Load predictions
        if preds_path:
            predictions = torch.load(preds_path, weights_only=False)
        else:
            # Try to find predictions in output directory
            pred_path = output_dir / f"{tomo_id}_pred.pt"
            if not pred_path.exists():
                raise FileNotFoundError("Could not find predictions file")
            predictions = torch.load(pred_path, weights_only=False)
        
        # Convert predictions to numpy array and ensure float32
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions, dtype=np.float32)
        elif predictions.dtype != np.float32:
            predictions = predictions.astype(np.float32)
        
        # Ensure input_data is 3D (Z,Y,X)
        if input_data.ndim > 3:
            input_data = input_data.squeeze()
        if input_data.ndim != 3:
            raise ValueError(f"Input data must be 3D after squeezing. Got shape: {input_data.shape}")
        
        # Ensure predictions is 3D
        if predictions.ndim > 3:
            predictions = predictions.squeeze()
        if predictions.ndim != 3:
            raise ValueError(f"Predictions must be 3D after squeezing. Got shape: {predictions.shape}")
        
        # Verify shapes match
        if input_data.shape != predictions.shape:
            # Try to match shapes if they're different
            if len(input_data.shape) == len(predictions.shape):
                predictions = predictions.reshape(input_data.shape)
            else:
                raise ValueError(f"Input shape {input_data.shape} does not match predictions shape {predictions.shape}")
        
        # Create model input tensor if model is provided
        if model is not None:
            model_input = torch.from_numpy(input_data).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            model_input = None
        
        # Create visualizations
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic visualizations
        create_slice_visualization(input_data, vis_dir / 'input_slices.png')
        create_slice_visualization(predictions, vis_dir / 'prediction_slices.png')
        # 3D surface and intensity profile generation removed per user request
        
        # Model-specific visualizations
        if model is None or model_input is None:
            return

        # Helper to run visualization for a single model object and save into a subfolder
        def _run_for_model(single_model, out_subdir: Path):
            fe = FeatureExtractor(single_model)
            with torch.no_grad():
                if torch.cuda.is_available():
                    _mi = model_input.cuda()
                    single_model = single_model.cuda()
                else:
                    _mi = model_input
                _ = single_model(_mi)

            out_subdir.mkdir(parents=True, exist_ok=True)
            create_encoder_decoder_flow(fe, input_data, predictions, out_subdir / 'encoder_decoder_flow.png')
            create_detailed_feature_analysis(fe, out_subdir)

        # If model is a list/tuple (ensemble)
        if isinstance(model, (list, tuple)):
            if iterate_ensemble:
                # Run for each member and create per-model subfolders
                for idx, m in enumerate(model):
                    sub = vis_dir / f'model_{idx}'
                    _run_for_model(m, sub)
            else:
                # Pick a single index if provided, otherwise default to first
                sel = 0 if model_index is None else int(model_index)
                if sel < 0 or sel >= len(model):
                    raise IndexError(f"model_index {sel} out of range for ensemble of size {len(model)}")
                _run_for_model(model[sel], vis_dir)
        else:
            # single model object
            _run_for_model(model, vis_dir)
        
    except Exception as e:
        raise RuntimeError(f"Visualization generation failed: {str(e)}") from e