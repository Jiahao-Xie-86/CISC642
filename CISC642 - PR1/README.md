# Image Processing and Mosaicking Project

This project implements various image processing techniques including image transformations (affine and perspective) and image mosaicking. It consists of two main parts:
- Part A: Image transformation using correspondence points
- Part B: Image mosaicking using pyramid representations and feature detection

## Prerequisites

- Python 3.9 or higher
- Required libraries:
  - OpenCV (cv2) 4.11.0
  - NumPy 1.26.4
  - Pandas 1.5.3
  - Matplotlib 3.7.0

## Installation

1. Install requirement:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── Template/
│   ├── PartA/
│   │   ├── original/           # Original images for transformation
│   │   ├── correspondances/    # Point correspondences for transformation
│   │   │   ├── affine/        # Points for affine transformation
│   │   │   └── perspective/    # Points for perspective transformation
│   │   └── outputs/           # Transformed image outputs
│   └── PartB/
│       ├── output/            # Manual mosaic outputs
│       ├── output_auto/       # Automatic mosaic outputs (extra credit)
│       └── image_left.png/image_right.png       # Source images for mosaicking
├── part_a.py             # Implementation of image transformations
├── part_b.py             # Implementation of image mosaicking
└── requirements.txt          # Project dependencies
```

## Part A: Image Transformations

### Running the Code
```bash
python Template/part_a.py
```

### Outputs
All outputs will be saved in `Template/PartA/outputs/`:

1. Transformed Images:
   - Affine Transformation:
     - `{image_name}_affine_method_a.png`: Using minimal correspondences
     - `{image_name}_affine_method_b.png`: Using all correspondences
   - Perspective Transformation:
     - `{image_name}_perspective_method_a.png`: Using minimal correspondences
     - `{image_name}_perspective_method_b.png`: Using all correspondences

2. Transformation Results:
   - `{image_name}_transformation_results.txt`: Contains
     - Transformation matrices
     - Error comparisons between methods

3. Additional analysis for `computer.jpg` with 10 extra correspondence points (labeled as 'computer_add_10')  



## Part B: Image Mosaicking

### Running the Code
```bash
python Template/part_b.py
```

### Outputs

#### 1. Manual Mosaicking (`Template/PartB/output/`)
- Image Processing Steps:
  - `convolved.png`: Convolution results
  - `reduced.png`: Image reduction
  - `expanded.png`: Image expansion

- Pyramid Representations:
  - `gaussian_pyramid_level_{i}.png`: Gaussian pyramid at level i
  - `laplacian_pyramid_level_{i}.png`: Laplacian pyramid at level i
  - `reconstructed.png`: Image reconstructed from Laplacian pyramid
  - `reconstruction_error.txt`: MSE and RMSE of reconstruction

- Mosaic Results:
  - `mosaic_pair{i}_{method}.png`: Mosaics with different transformations
  - `mosaic_pair{i}_none.png`: Basic mosaics without transformation

#### 2. Automatic Mosaicking (`Template/PartB/output_auto/`)
- `auto_mosaic_pair{i}.png`: Automatically generated mosaics (extra credit feature)

## Important Notes

1. Part A Requirements:
   - Ensure all correspondence point CSV files are present in their respective directories
   - Points should be properly formatted in the CSV files

2. Part B Requirements:
   - Manual mosaicking requires user interaction to:
     - Select corresponding points
     - Define blend boundaries
   - Automatic mosaicking success depends on the quality of detected feature points

3. Error Handling:
   - The program will create output directories if they don't exist
   - Error messages will be displayed if required files are missing


For more detailed information about the implementation, refer to the comments in the source code files. 