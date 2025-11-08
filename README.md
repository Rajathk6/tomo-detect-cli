# Tomo-Detect

A command-line tool for detecting motor coordinates in tomography data using deep learning models.

Check out the project details and installation on PYPI: 
   > [PYPI guide]{https://pypi.org/project/tomo-detect/0.1.7/}
## Installation

```bash
pip install tomo-detect
```

## Usage

Basic usage:

```bash
tomo-detect input_path_A
tomo-detect input_path_A input_path_B
```

The input path can be either:

- A .npy file containing tomography data
- A zip file containing multiple tomography files
- A directory containing .npy or .mrc files

### Options

```bash
tomo-detect --help                        # Show help message and usage information
tomo-detect input.npy --debug             # Enable debug logging
tomo-detect input.zip --test              # Run test mode with additional validations
tomo-detect input.npy --output path       # Specify custom output directory
tomo-detect input.npy --detailed          # Outputs visualizations for the inputs
tomo-detect input.npy --batch-size 4      # Set custom batch size for inference
tomo-detect input.npy --device cpu/gpu    # Force CPU inference

```

### Output Files

The tool generates several output files:

1. `motor_detections_submission.csv` - Contains motor coordinates (primary output)
2. `motor_detections_detailed.csv` - Includes additional detection information
3. `predictions.npy` - Raw probability maps
4. `masks.npy` - Binary masks derived from predictions
5. `summary.json` - Detection summary and statistics

### Example

```bash
# Process a single .npy file
tomo-detect sample.npy

# Process multiple files in a zip
tomo-detect samples.zip

# Enable debug mode for detailed logging
tomo-detect sample.npy --debug
```

## Tips and Tricks

1. **Input Preparation**:

   - Ensure input files are properly formatted numpy arrays
   - For .mrc files, they will be automatically converted
   - Zip files should contain only supported file types
   - Supports multiple inputs and enbales parallel processing based on the system specs

2. **Performance Optimization**:

   - Use GPU acceleration when available
   - Adjust batch size based on your memory capacity
   - For large datasets, consider processing in chunks
   - It is advised to provide unzipped files so as to not bottleneck the cpu performance
   - Run it using --test to confirm the working of the model and save time

3. **Troubleshooting**:
   - Enable --debug mode for detailed logging
   - Check system requirements before running
   - Verify input file formats and dimensions
   - --detailed consumes lot of cpu memory

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- Minimum 8GB RAM
- 2GB disk space for models

## License

MIT License
