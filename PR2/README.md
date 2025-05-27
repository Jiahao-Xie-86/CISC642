# Stereo Analysis System

## Overview
This project implements a multi-resolution stereo analysis system for depth estimation from stereo image pairs. It combines region-based and feature-based matching methods to generate accurate disparity maps.

## Table of Contents
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Configuration Details](#configuration-details)
- [Implementation Details](#implementation-details)
- [Results](#results)

## Features
- **Multi-resolution Processing**
  - Configurable pyramid levels
  - Coarse-to-fine disparity estimation
  - Different matching methods at different levels

- **Matching Methods**
  - Region-based matching with:
    - Sum of Absolute Differences (SAD)
    - Sum of Squared Differences (SSD)
    - Normalized Cross-Correlation (NCC)
  - Feature-based matching using Harris corner detector

- **Advanced Processing**
  - Occlusion detection through left-right consistency checks
  - Adaptive gap filling
  - Configurable template sizes and search ranges

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Required Packages
```bash
pip install numpy opencv-python matplotlib scipy scikit-image
```

## Dataset Setup

### Download
The system uses the Middlebury stereo dataset (2001). Download from:
https://vision.middlebury.edu/stereo/data/scenes2001/

### Directory Structure
```
./images/
├── barn1/
│   ├── im2.ppm (left image)
│   └── im6.ppm (right image)
├── bull/
├── poster/
├── sawtooth/
└── venus/
```

## Usage

### Basic Usage
Run with default parameters:
```bash
python stereo_matching.py
```

### Advanced Usage
Run with custom parameters:
```bash
python stereo_matching.py \
    --image_dir ./images \
    --output_dir ./results \
    --stereo_pair barn1 \
    --levels 3 \
    --match_method NCC \
    --template_height 7 \
    --template_width 7 \
    --max_disparity 64 \
    --feature_based_levels 1 \
    --patch_size 11 \
    --search_range 5
```

### Batch Processing
Run multiple configurations:
```bash
python run_experiments.py
```


## Configuration Details

### Available Parameters
- `--image_dir`: Input image directory
- `--output_dir`: Output results directory
- `--stereo_pair`: Target stereo pair ('all', 'barn1', 'bull', 'poster', 'sawtooth', 'venus')
- `--levels`: Number of pyramid levels
- `--match_method`: Matching method ('SAD', 'SSD', 'NCC')
- `--template_height`: Template height for region-based matching
- `--template_width`: Template width for region-based matching
- `--max_disparity`: Maximum disparity value
- `--feature_based_levels`: Levels for feature-based matching (comma-separated)
- `--patch_size`: Patch size for feature-based matching
- `--search_range`: Search range around initial disparity

### Predefined Configurations
1. **Basic NCC Matching**
   - 3 levels, NCC matching
   - 7×7 template
   - No feature-based matching

2. **Multi-resolution with Features**
   - 4 levels, NCC matching
   - 5×5 template
   - Feature-based on level 1

3. **SSD Matching**
   - 5 levels, SSD matching
   - 5×5 template
   - No feature-based matching

4. **High Resolution**
   - 4 levels, NCC matching
   - 5×5 template
   - No feature-based matching

5. **SAD with Rectangular Template**
   - 5 levels, SAD matching
   - 3×5 template
   - No feature-based matching

## Implementation Details

### Region-Based Matching
- Template matching with configurable window size
- Three scoring methods: SAD, SSD, NCC
- Disparity search within specified range

### Feature-Based Matching
- Harris corner detection
- Feature point matching using same metrics
- Interpolation for non-feature pixels

### Multi-Resolution Processing
- Image pyramid construction
- Coarse-to-fine disparity estimation
- Disparity initialization from coarser levels

### Post-Processing
- Left-right consistency check for occlusion detection
- Adaptive window-based gap filling
- Disparity map refinement

## Results

### Output Images
The system generates two types of disparity maps for each configuration:

1. **Left-to-Right Disparity Map**
   - Shows the displacement of pixels from the left image to the right image
   - Brighter pixels indicate larger disparities (objects closer to the camera)
   - Darker pixels indicate smaller disparities (objects further from the camera)
   - Black pixels indicate invalid matches or occlusions

2. **Right-to-Left Disparity Map**
   - Shows the displacement of pixels from the right image to the left image
   - Used for consistency checking with left-to-right results
   - Helps identify occluded regions and mismatches

### Filename Convention
Results are saved with filenames that encode all important parameters:
```
{stereo_pair}_lvl{levels}_meth{matching_method}_temp{template_height}x{template_width}_maxdisp{max_disparity}_{direction}.png
```

Example: `barn1_lvl3_methNCC_temp7x7_maxdisp64_left_to_right.png`
- `barn1`: Stereo pair name
- `lvl3`: 3 pyramid levels used
- `methNCC`: Normalized Cross-Correlation matching method
- `temp7x7`: 7×7 pixel template size
- `maxdisp64`: Maximum disparity of 64 pixels
- `left_to_right`: Direction of disparity computation

### Output Directory Structure
```
./results/
├── config_1/
│   ├── barn1_lvl3_methNCC_temp7x7_maxdisp64_left_to_right.png
│   ├── barn1_lvl3_methNCC_temp7x7_maxdisp64_right_to_left.png
│   ├── bull_lvl3_methNCC_temp7x7_maxdisp64_left_to_right.png
│   └── ...
├── config_2/
│   └── ...
└── ...
```

### Interpreting Results
- **Disparity Range**: The grayscale values in the output images represent disparity values:
  - 0 (black) = No match found or occlusion
  - 255 (white) = Maximum disparity (objects closest to camera)
  - Intermediate values = Proportional to depth

- **Quality Indicators**:
  - Smooth regions with consistent disparities indicate good matches
  - Sharp discontinuities often indicate depth boundaries
  - Black regions (0) indicate occlusions or failed matches
  - Noisy or inconsistent regions may indicate matching errors

- **Comparing Configurations**:
  - Higher pyramid levels generally provide better results for complex scenes
  - NCC typically gives more robust results than SAD or SSD
  - Feature-based matching can improve results in textureless regions
  - Larger templates provide more stable results but may blur depth boundaries

