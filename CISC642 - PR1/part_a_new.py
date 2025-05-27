import os
import cv2 as cv
import numpy as np
import pandas as pd

def load_correspondences(csv_path):
    """Loads corresponding points from a CSV file (skipping the header)."""
    data = pd.read_csv(csv_path)
    return data.values  # Convert DataFrame to numpy array

def normalize_points(pts):
    """
    Normalize points by subtracting the centroid and scaling so that the mean distance
    from the origin is sqrt(2). Returns the normalized points and the normalization matrix T.
    
    Args:
        pts: Input points as numpy array
    Returns:
        pts_norm: Normalized points
        T: Normalization matrix
    """
    pts = pts.astype(np.float32)
    # Calculate centroid of points
    centroid = np.mean(pts, axis=0)
    # Center points around origin
    pts_centered = pts - centroid
    # Calculate average distance from origin
    avg_dist = np.mean(np.linalg.norm(pts_centered, axis=1))
    # Scale points so mean distance is sqrt(2)
    scale = np.sqrt(2) / avg_dist if avg_dist > 0 else 1.0
    # Create normalization matrix
    T = np.array([[scale,     0, -scale * centroid[0]],
                  [    0, scale, -scale * centroid[1]],
                  [    0,     0,                  1]], dtype=np.float32)
    # Convert to homogeneous coordinates and apply normalization
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = pts_norm_h[:, :2]
    return pts_norm, T

def get_affine_transform_method_a(src_pts, dst_pts):
    """
    Estimate the affine transformation (2x3) using the minimal 3 correspondences.
    Both source and destination points are normalized before solving and then denormalized.
    
    Args:
        src_pts: Source points (original image points)
        dst_pts: Destination points (transformed image points)
    Returns:
        A_est_full[:2, :]: 2x3 affine transformation matrix
    """
    # Normalize minimal 3 points from each set
    src_norm, T_src = normalize_points(src_pts[:3])
    dst_norm, T_dst = normalize_points(dst_pts[:3])
    
    # Build system of equations for affine transformation
    # Each point pair gives 2 equations for the 6 unknown parameters
    A = []
    B = []
    for i in range(3):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(u)
        B.append(v)
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    # Solve the system using least squares
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    A_norm = X.reshape(2, 3)
    
    # Denormalize the transformation matrix
    A_norm_full = np.vstack([A_norm, [0, 0, 1]])
    A_est_full = np.linalg.inv(T_dst) @ A_norm_full @ T_src
    return A_est_full[:2, :]

def get_affine_transform_method_b(src_pts, dst_pts):
    """
    Estimate the affine transformation using an over-constrained set of correspondences.
    This method is more robust to noise as it uses all available point pairs.
    
    Args:
        src_pts (numpy.ndarray): Source points (original image points)
        dst_pts (numpy.ndarray): Destination points (transformed image points)
    Returns:
        numpy.ndarray: 2x3 affine transformation matrix
    """
    # Normalize all points from each set
    src_norm, T_src = normalize_points(src_pts)
    dst_norm, T_dst = normalize_points(dst_pts)
    
    num_pts = src_norm.shape[0]
    A = []
    B = []
    for i in range(num_pts):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(u)
        B.append(v)
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    A_norm = X.reshape(2, 3)
    
    A_norm_full = np.vstack([A_norm, [0, 0, 1]])
    A_est_full = np.linalg.inv(T_dst) @ A_norm_full @ T_src
    return A_est_full[:2, :]

def get_perspective_transform_method_a(src_pts, dst_pts):
    """
    Estimate the perspective transformation matrix (3x3) using the minimal 4 correspondences.
    Points are normalized before solving, then the transformation is denormalized.
    
    Args:
        src_pts: Source points (original image points)
        dst_pts: Destination points (transformed image points)
    Returns:
        H_est: 3x3 perspective transformation matrix
    """
    # Normalize minimal 4 points
    src_norm, T_src = normalize_points(src_pts[:4])
    dst_norm, T_dst = normalize_points(dst_pts[:4])
    
    # Build system of equations for perspective transformation
    # Each point pair gives 2 equations for the 8 unknown parameters
    A = []
    B = []
    for i in range(4):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        B.append(u)
        B.append(v)
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    # Solve the system using least squares
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    H_norm = np.append(X, 1).reshape(3, 3)
    # Denormalize the transformation matrix
    H_est = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H_est / H_est[2, 2]  # Normalize by last element

def get_perspective_transform_method_b(src_pts, dst_pts):
    """
    Estimate the perspective transformation using an over-constrained set of correspondences.
    This method is more robust to noise as it uses all available point pairs.
    
    Args:
        src_pts (numpy.ndarray): Source points (original image points)
        dst_pts (numpy.ndarray): Destination points (transformed image points)
    Returns:
        numpy.ndarray: 3x3 perspective transformation matrix
    """
    src_norm, T_src = normalize_points(src_pts)
    dst_norm, T_dst = normalize_points(dst_pts)
    
    num_pts = src_norm.shape[0]
    A = []
    B = []
    for i in range(num_pts):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        B.append(u)
        B.append(v)
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    H_norm = np.append(X, 1).reshape(3, 3)
    H_est = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H_est / H_est[2, 2]

def calculate_error(estimated, builtin):
    """Compute the Frobenius norm error between estimated and built-in matrices."""
    return np.linalg.norm(estimated - builtin)

def warp_affine(image, matrix):
    """Apply an affine transformation to an image."""
    h, w = image.shape[:2]
    return cv.warpAffine(image, matrix, (w, h))

def warp_perspective(image, matrix):
    """Apply a perspective transformation to an image."""
    h, w = image.shape[:2]
    return cv.warpPerspective(image, matrix, (w, h))

# Get absolute paths for input and output directories
current_dir = os.path.dirname(os.path.abspath(__file__))
original_dir = os.path.join(current_dir, "PartA", "original")
outputs_dir = os.path.join(current_dir, "PartA", "outputs")
affine_dir = os.path.join(current_dir, "PartA", "correspondances", "affine")
perspective_dir = os.path.join(current_dir, "PartA", "correspondances", "perspective")

# Create outputs directory if it doesn't exist
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

# Define input files and their extensions for our five images
image_files = {
    "computer": "png",
    "lena": "png",
    "mario": "jpg",
    "mountain": "jpg",
    "water": "jpg",
    "computer_add_10": "png"
}

print(f"Looking for images in: {original_dir}")
print(f"Looking for affine points in: {affine_dir}")
print(f"Looking for perspective points in: {perspective_dir}")
print(f"Output directory: {outputs_dir}")

# Process each image
for file_name, extension in image_files.items():
    print(f"\nProcessing {file_name}.{extension}")
    
    # Construct full paths for image and correspondence CSV files
    image_path = os.path.join(original_dir, f"{file_name}.{extension}")
    affine_path = os.path.join(affine_dir, f"{file_name}.csv")
    perspective_path = os.path.join(perspective_dir, f"{file_name}.csv")
    
    print(f"Reading image from: {image_path}")
    print(f"Reading affine points from: {affine_path}")
    print(f"Reading perspective points from: {perspective_path}")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        continue
    if not os.path.exists(affine_path):
        print(f"Affine points file not found: {affine_path}")
        continue
    if not os.path.exists(perspective_path):
        print(f"Perspective points file not found: {perspective_path}")
        continue
        
    # Read the image
    original_image = cv.imread(image_path)
    if original_image is None:
        print(f"Failed to read image: {image_path}")
        continue
    print(f"Successfully loaded image: {file_name}.{extension}")

    # Read the correspondence points
    try:
        affine_data = load_correspondences(affine_path)
        perspective_data = load_correspondences(perspective_path)
        print("Successfully loaded correspondence points")
    except Exception as e:
        print(f"Error reading correspondence files: {str(e)}")
        continue

    # For the 'computer' image, check if extra correspondences exist
    if file_name == "computer":
        extra_affine_path = os.path.join(affine_dir, f"{file_name}_extra.csv")
        extra_perspective_path = os.path.join(perspective_dir, f"{file_name}_extra.csv")
        if os.path.exists(extra_affine_path):
            try:
                extra_affine_data = load_correspondences(extra_affine_path)
                affine_data = np.vstack([affine_data, extra_affine_data])
                print("Extra affine correspondences added.")
            except Exception as e:
                print(f"Error reading extra affine correspondences: {str(e)}")
        if os.path.exists(extra_perspective_path):
            try:
                extra_perspective_data = load_correspondences(extra_perspective_path)
                perspective_data = np.vstack([perspective_data, extra_perspective_data])
                print("Extra perspective correspondences added.")
            except Exception as e:
                print(f"Error reading extra perspective correspondences: {str(e)}")

    # Extract original and transformed points.
    # For affine: columns [x_og, y_og] and [x_aff, y_aff]
    original_points = affine_data[:, :2]
    affine_points = np.column_stack((affine_data[:, 2], affine_data[:, 3]))
    # For perspective: use perspective CSV: columns [x_og, y_og] and [x_per, y_per]
    perspective_points = np.column_stack((perspective_data[:, 0], perspective_data[:, 1]))
    
    # Convert to float32 for numerical stability
    original_points = original_points.astype(np.float32)
    affine_points = affine_points.astype(np.float32)
    perspective_points = perspective_points.astype(np.float32)
    
    # Get built-in transformation matrices using minimal correspondences
    affine_transform_builtin = cv.getAffineTransform(original_points[:3], affine_points[:3])
    perspective_transform_builtin = cv.getPerspectiveTransform(original_points[:4], perspective_points[:4])
    
    # Apply our transformation methods
    # Method A: minimal number of correspondences
    affine_transform_method_a = get_affine_transform_method_a(original_points, affine_points)
    affine_image_method_a = warp_affine(original_image, affine_transform_method_a)

    perspective_transform_method_a = get_perspective_transform_method_a(original_points, perspective_points)
    perspective_image_method_a = warp_perspective(original_image, perspective_transform_method_a)
    
    # Method B: over-constrained (all correspondences)
    affine_transform_method_b = get_affine_transform_method_b(original_points, affine_points)
    affine_image_method_b = warp_affine(original_image, affine_transform_method_b)

    perspective_transform_method_b = get_perspective_transform_method_b(original_points, perspective_points)
    perspective_image_method_b = warp_perspective(original_image, perspective_transform_method_b)
    
    # Compute errors between our estimates and OpenCV's built-in functions
    affine_error_method_a = calculate_error(affine_transform_method_a, affine_transform_builtin)
    affine_error_method_b = calculate_error(affine_transform_method_b, affine_transform_builtin)
    perspective_error_method_a = calculate_error(perspective_transform_method_a, perspective_transform_builtin)
    perspective_error_method_b = calculate_error(perspective_transform_method_b, perspective_transform_builtin)
    
    # Prepare results text for output
    results_text = []
    results_text.append('############################################################')
    results_text.append(f'Processed image: {file_name}')
    results_text.append('------------------------------------------------------------')
    results_text.append('Affine transformation matrix (Method A - minimal correspondences):')
    results_text.append(f'{affine_transform_method_a}')
    results_text.append('------------------------------------------------------------')
    results_text.append('Affine transformation matrix (Method B - over-constrained):')
    results_text.append(f'{affine_transform_method_b}')
    results_text.append('------------------------------------------------------------')
    results_text.append('Perspective transformation matrix (Method A - minimal correspondences):')
    results_text.append(f'{perspective_transform_method_a}')
    results_text.append('------------------------------------------------------------')
    results_text.append('Perspective transformation matrix (Method B - over-constrained):')
    results_text.append(f'{perspective_transform_method_b}')
    results_text.append('------------------------------------------------------------')
    results_text.append(f'Error (Affine Method A vs. built-in): {affine_error_method_a}')
    results_text.append(f'Error (Affine Method B vs. built-in): {affine_error_method_b}')
    results_text.append(f'Error (Perspective Method A vs. built-in): {perspective_error_method_a}')
    results_text.append(f'Error (Perspective Method B vs. built-in): {perspective_error_method_b}')
    results_text.append('############################################################\n')
    
    # Print results to console
    for line in results_text:
        print(line)
    
    # Save results to a text file
    results_file = os.path.join(outputs_dir, f"{file_name}_transformation_results.txt")
    with open(results_file, 'w') as f:
        f.write('\n'.join(results_text))
    print(f"Results saved to: {results_file}")
    
    # Save transformed images
    try:
        output_base = os.path.join(outputs_dir, file_name)
        cv.imwrite(f"{output_base}_affine_method_a.png", affine_image_method_a)
        cv.imwrite(f"{output_base}_affine_method_b.png", affine_image_method_b)
        cv.imwrite(f"{output_base}_perspective_method_a.png", perspective_image_method_a)
        cv.imwrite(f"{output_base}_perspective_method_b.png", perspective_image_method_b)
        print(f"Saved transformed images for {file_name}")
    except Exception as e:
        print(f"Error saving transformed images: {str(e)}")
    
    print(f"Completed processing {file_name}")
