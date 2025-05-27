
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import argparse
import os
from typing import Tuple, List, Optional, Dict, Any


def read_images(image_dir: str, stereo_pair: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the left and right images of a stereo pair.
    
    Args:
        image_dir: Directory containing the stereo image datasets
        stereo_pair: Name of the stereo pair (e.g., 'barn1', 'bull', etc.)
    
    Returns:
        Tuple containing left and right grayscale images
    """
    left_img = cv2.imread(os.path.join(image_dir, stereo_pair, 'im2.ppm'), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(os.path.join(image_dir, stereo_pair, 'im6.ppm'), cv2.IMREAD_GRAYSCALE)
    
    if left_img is None or right_img is None:
        raise ValueError(f"Failed to load images for {stereo_pair}")
    
    return left_img, right_img


def compute_sad(template: np.ndarray, window: np.ndarray) -> float:
    """
    Compute Sum of Absolute Differences (SAD) between template and window.
    
    Args:
        template: Template patch
        window: Window patch to match against
    
    Returns:
        SAD score (lower is better match)
    """
    return np.sum(np.abs(template - window))


def compute_ssd(template: np.ndarray, window: np.ndarray) -> float:
    """
    Compute Sum of Squared Differences (SSD) between template and window.
    
    Args:
        template: Template patch
        window: Window patch to match against
    
    Returns:
        SSD score (lower is better match)
    """
    return np.sum((template - window) ** 2)


def compute_ncc(template: np.ndarray, window: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation (NCC) between template and window.
    
    Args:
        template: Template patch
        window: Window patch to match against
    
    Returns:
        NCC score (higher is better match)
    """
    # Ensure we don't divide by zero
    template_flat = template.flatten() - np.mean(template)
    window_flat = window.flatten() - np.mean(window)
    
    template_norm = np.sqrt(np.sum(template_flat ** 2))
    window_norm = np.sqrt(np.sum(window_flat ** 2))
    
    # Avoid division by zero
    if template_norm == 0 or window_norm == 0:
        return 0
    
    return np.sum(template_flat * window_flat) / (template_norm * window_norm)


def get_matching_score(template: np.ndarray, window: np.ndarray, method: str) -> float:
    """
    Compute matching score based on specified method.
    
    Args:
        template: Template patch
        window: Window patch to match against
        method: Matching method ('SAD', 'SSD', or 'NCC')
    
    Returns:
        Matching score (for SAD and SSD, lower is better; for NCC, higher is better)
    """
    if method == 'SAD':
        return compute_sad(template, window)
    elif method == 'SSD':
        return compute_ssd(template, window)
    elif method == 'NCC':
        return compute_ncc(template, window)
    else:
        raise ValueError(f"Unknown matching method: {method}")


def region_based_matching(left_img: np.ndarray, right_img: np.ndarray, 
                         template_size: Tuple[int, int], match_method: str,
                         max_disparity: int, disp_init: Optional[np.ndarray] = None,
                         search_range: int = 3) -> np.ndarray:
    """
    Perform region-based stereo matching.
    
    Args:
        left_img: Left image (reference)
        right_img: Right image
        template_size: Size of template window (height, width)
        match_method: Matching method ('SAD', 'SSD', or 'NCC')
        max_disparity: Maximum disparity to search for
        disp_init: Initial disparity map (for initialization)
        search_range: Range to search around initial disparity
    
    Returns:
        Disparity map
    """
    height, width = left_img.shape
    half_h, half_w = template_size[0] // 2, template_size[1] // 2
    
    # Initialize disparity map
    disparity = np.zeros((height, width), dtype=np.float32)
    
    # Pad images for boundary handling
    left_padded = np.pad(left_img, ((half_h, half_h), (half_w, half_w)), mode='constant')
    right_padded = np.pad(right_img, ((half_h, half_h), (half_w, half_w)), mode='constant')
    
    # For each pixel in left image
    for y in range(half_h, height + half_h):
        for x in range(half_w, width + half_w):
            # Skip boundaries
            if x < half_w + 1 or x >= width + half_w - 1:
                continue
                
            # Extract template from left image
            template = left_padded[y - half_h:y + half_h + 1, x - half_w:x + half_w + 1]
            
            # Determine search range based on initial disparity if available
            if disp_init is not None:
                # Get initial disparity for this pixel
                init_disp = disp_init[y - half_h, x - half_w]
                min_d = max(0, int(init_disp) - search_range)
                max_d = min(max_disparity, int(init_disp) + search_range)
            else:
                # Without initialization, search full range
                min_d = 0
                max_d = max_disparity
            
            best_score = float('inf') if match_method in ['SAD', 'SSD'] else float('-inf')
            best_disp = 0
            
            # Search for best matching window in right image
            for d in range(min_d, max_d + 1):
                # Skip if out of bounds
                if x - d < half_w or x - d >= width + half_w:
                    continue
                    
                # Extract window from right image
                window = right_padded[y - half_h:y + half_h + 1, (x - d) - half_w:(x - d) + half_w + 1]
                
                # Compute matching score
                score = get_matching_score(template, window, match_method)
                
                # Update best match
                if (match_method in ['SAD', 'SSD'] and score < best_score) or \
                   (match_method == 'NCC' and score > best_score):
                    best_score = score
                    best_disp = d
            
            # Store disparity
            disparity[y - half_h, x - half_w] = best_disp
    
    return disparity


def harris_corner_detector(image: np.ndarray, block_size: int = 2, 
                          aperture_size: int = 3, k: float = 0.04) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect Harris corners in an image.
    
    Args:
        image: Input grayscale image
        block_size: Size of neighborhood for corner detection
        aperture_size: Aperture parameter for Sobel operator
        k: Harris detector free parameter
    
    Returns:
        Tuple containing corner response map and corner locations
    """
    # Compute Harris corner response map
    corner_response = cv2.cornerHarris(image.astype(np.float32), block_size, aperture_size, k)
    
    # Normalize response for visualization
    corner_response_norm = cv2.normalize(corner_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Find corner points (non-maximum suppression)
    _, corners = cv2.threshold(corner_response_norm, 0.01 * corner_response_norm.max(), 255, 0)
    corners = np.where(corners > 0)
    
    return corner_response, np.array(list(zip(corners[1], corners[0])))  # x, y format


def extract_patch(image, point, patch_size):
    """
    Extract a patch centered at a point with proper boundary handling.
    
    Args:
        image: Input image
        point: Center point (x, y)
        patch_size: Size of the patch
    
    Returns:
        Extracted patch
    """
    import numpy as np
    
    x, y = point
    half_size = patch_size // 2
    
    # Get image dimensions
    height, width = image.shape
    
    # Create full-sized patch filled with zeros
    full_patch = np.zeros((patch_size, patch_size), dtype=image.dtype)
    
    # Calculate boundaries for extraction from the image
    y_min = max(0, y - half_size)
    y_max = min(height, y + half_size + 1)
    x_min = max(0, x - half_size)
    x_max = min(width, x + half_size + 1)
    
    # Calculate boundaries for placement in the full-sized patch
    dest_y_min = max(0, half_size - (y - y_min))
    dest_y_max = min(patch_size, dest_y_min + (y_max - y_min))
    dest_x_min = max(0, half_size - (x - x_min))
    dest_x_max = min(patch_size, dest_x_min + (x_max - x_min))
    
    # Extract patch from image
    src_patch = image[y_min:y_max, x_min:x_max]
    
    # Check dimensions
    h_extract, w_extract = src_patch.shape
    h_dest = dest_y_max - dest_y_min
    w_dest = dest_x_max - dest_x_min
    
    # Ensure destination area and source patch have the same dimensions
    h_to_copy = min(h_extract, h_dest)
    w_to_copy = min(w_extract, w_dest)
    
    # Place extracted patch in the full-sized patch
    if h_to_copy > 0 and w_to_copy > 0:
        full_patch[dest_y_min:dest_y_min+h_to_copy, dest_x_min:dest_x_min+w_to_copy] = \
            src_patch[:h_to_copy, :w_to_copy]
    
    return full_patch


def feature_based_matching(left_img, right_img, patch_size, match_method, max_disparity):
    """
    Perform feature-based stereo matching using Harris corners.
    
    Args:
        left_img: Left image (reference)
        right_img: Right image
        patch_size: Size of patch around feature points
        match_method: Matching method ('SAD', 'SSD', or 'NCC')
        max_disparity: Maximum disparity to search for
    
    Returns:
        Disparity map
    """
    import numpy as np
    import cv2
    from scipy.ndimage import gaussian_filter
    
    height, width = left_img.shape
    
    # Initialize disparity map
    disparity = np.zeros((height, width), dtype=np.float32)
    
    # Safety margin for boundary handling
    margin = patch_size // 2 + 1
    
    try:
        # Detect Harris corners in the left image
        # First convert to float32
        left_float = left_img.astype(np.float32)
        
        # Harris parameters
        block_size = 2
        aperture_size = 3
        k = 0.04
        
        # Compute Harris corner response
        corner_response = cv2.cornerHarris(left_float, block_size, aperture_size, k)
        
        # Threshold for corner detection
        corner_threshold = 0.01 * corner_response.max()
        
        # Find corner points
        corners = np.where(corner_response > corner_threshold)
        corner_points = list(zip(corners[1], corners[0]))  # x, y format
        
        # Filter corners - remove those too close to the boundary
        filtered_corners = []
        for x, y in corner_points:
            if margin <= x < width - margin and margin <= y < height - margin:
                filtered_corners.append((x, y))
                
        # Limit the number of corners to avoid excessive computation
        if len(filtered_corners) > 1000:
            # Sort corners by response strength
            sorted_corners = sorted(
                filtered_corners, 
                key=lambda pt: corner_response[pt[1], pt[0]], 
                reverse=True
            )
            filtered_corners = sorted_corners[:1000]
        
        # If no valid corners found, use a grid of points
        if len(filtered_corners) < 100:
            grid_step = max(5, min(width, height) // 50)
            for y in range(margin, height - margin, grid_step):
                for x in range(margin, width - margin, grid_step):
                    filtered_corners.append((x, y))
        
        print(f"  Using {len(filtered_corners)} feature points")
        
        # Extract patches and compute matching
        for x, y in filtered_corners:
            # Extract patch from left image
            left_patch = extract_patch(left_img, (x, y), patch_size)
            
            best_score = float('inf') if match_method in ['SAD', 'SSD'] else float('-inf')
            best_disp = 0
            
            # Search for matching patch in right image
            # Limit search to valid disparity range to avoid boundary issues
            max_d = min(x - margin, max_disparity)
            
            for d in range(0, max_d + 1):
                # Extract patch from right image
                right_patch = extract_patch(right_img, (x - d, y), patch_size)
                
                # Compute matching score
                if match_method == 'SAD':
                    score = np.sum(np.abs(left_patch - right_patch))
                elif match_method == 'SSD':
                    score = np.sum((left_patch - right_patch) ** 2)
                else:  # NCC
                    # Normalize patches
                    left_norm = left_patch - np.mean(left_patch)
                    right_norm = right_patch - np.mean(right_patch)
                    
                    # Avoid division by zero
                    left_std = np.std(left_patch)
                    right_std = np.std(right_patch)
                    
                    if left_std == 0 or right_std == 0:
                        score = 0
                    else:
                        score = np.sum(left_norm * right_norm) / (left_std * right_std * patch_size * patch_size)
                
                # Update best match
                if (match_method in ['SAD', 'SSD'] and score < best_score) or \
                   (match_method == 'NCC' and score > best_score):
                    best_score = score
                    best_disp = d
            
            # Store disparity
            disparity[y, x] = best_disp
        
        # Spread sparse disparities to create a dense disparity map
        # First, apply Gaussian smoothing
        smoothed = gaussian_filter(disparity, sigma=5)
        
        # Create a mask of sparse disparity points
        mask = disparity > 0
        
        # Combine original disparities with smoothed values
        # Keep original where they exist, use smoothed elsewhere
        disparity = np.where(mask, disparity, smoothed)
        
    except Exception as e:
        print(f"  Error in feature matching: {e}")
        # Return sparse disparity map on error
        
    return disparity


def perform_validity_check(left_to_right: np.ndarray, right_to_left: np.ndarray, 
                          threshold: float = 1.0) -> np.ndarray:
    """
    Perform validity check by comparing left-to-right and right-to-left disparity maps.
    
    Args:
        left_to_right: Disparity map from left to right matching
        right_to_left: Disparity map from right to left matching
        threshold: Threshold for disparity consistency
    
    Returns:
        Valid disparity map (with invalid areas set to 0)
    """
    height, width = left_to_right.shape
    valid_disparity = np.zeros_like(left_to_right)
    
    for y in range(height):
        for x in range(width):
            # Get disparity from left to right
            d = left_to_right[y, x]
            
            # Check if the corresponding pixel in the right image maps back to this pixel
            x_right = max(0, min(width - 1, int(x - d)))
            d_right = right_to_left[y, x_right]
            
            # Validate consistency
            if abs(d - d_right) <= threshold:
                valid_disparity[y, x] = d
    
    return valid_disparity


def fill_gaps(disparity: np.ndarray) -> np.ndarray:
    """
    Fill gaps in disparity map by averaging valid disparities in neighborhood.
    
    Args:
        disparity: Input disparity map with gaps (zeros)
    
    Returns:
        Filled disparity map
    """
    filled_disparity = disparity.copy()
    height, width = disparity.shape
    
    # Find gap locations
    gaps = np.where(disparity == 0)
    gap_coords = list(zip(gaps[0], gaps[1]))
    
    # Fill each gap
    for y, x in gap_coords:
        # Start with a 3x3 window and expand if needed
        window_size = 3
        valid_values = []
        
        while len(valid_values) < 5 and window_size <= 11:
            half_size = window_size // 2
            
            # Define window boundaries
            y_min = max(0, y - half_size)
            y_max = min(height, y + half_size + 1)
            x_min = max(0, x - half_size)
            x_max = min(width, x + half_size + 1)
            
            # Extract valid values from window
            window = disparity[y_min:y_max, x_min:x_max]
            valid_values = window[window > 0]
            
            # If not enough valid values, increase window size
            if len(valid_values) < 5:
                window_size += 2
        
        # Fill gap with average of valid values if available
        if len(valid_values) > 0:
            filled_disparity[y, x] = np.mean(valid_values)
    
    return filled_disparity


def create_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Create an image pyramid with specified number of levels.
    
    Args:
        image: Input image
        levels: Number of pyramid levels
    
    Returns:
        List of images at different resolutions
    """
    pyramid = [image]
    
    for i in range(1, levels):
        # Downsample image for next level
        h, w = pyramid[-1].shape
        new_h, new_w = h // 2, w // 2
        downsampled = resize(pyramid[-1], (new_h, new_w), anti_aliasing=True)
        pyramid.append(downsampled)
    
    # Reverse to have coarsest resolution first
    return pyramid[::-1]


def multi_resolution_stereo(left_img: np.ndarray, right_img: np.ndarray,
                           levels: int, template_sizes: List[Tuple[int, int]],
                           match_methods: List[str], max_disparities: List[int],
                           feature_levels: List[int], patch_size: int,
                           search_ranges: List[int]) -> np.ndarray:
    """
    Perform multi-resolution stereo matching.
    
    Args:
        left_img: Left image
        right_img: Right image
        levels: Number of pyramid levels
        template_sizes: List of template sizes for each level
        match_methods: List of matching methods for each level
        max_disparities: List of maximum disparities for each level
        feature_levels: List of levels where feature-based matching should be used
        patch_size: Patch size for feature-based matching
        search_ranges: List of search ranges for each level
    
    Returns:
        Final disparity map
    """
    # Create image pyramids
    left_pyramid = create_pyramid(left_img, levels)
    right_pyramid = create_pyramid(right_img, levels)
    
    # Start with the coarsest level (must use region-based at top level)
    disp_init = None
    
    # Process each level
    for i in range(levels):
        current_left = left_pyramid[i]
        current_right = right_pyramid[i]
        
        # Determine if we should use feature-based matching at this level
        use_feature_based = i in feature_levels
        
        # For the coarsest level, always use region-based
        if i == 0:
            use_feature_based = False
        
        # Compute disparities from left to right
        if use_feature_based:
            l_to_r_disp = feature_based_matching(current_left, current_right, 
                                                patch_size, match_methods[i], 
                                                max_disparities[i])
        else:
            l_to_r_disp = region_based_matching(current_left, current_right, 
                                              template_sizes[i], match_methods[i], 
                                              max_disparities[i], disp_init,
                                              search_ranges[i])
        
        # Compute disparities from right to left
        if use_feature_based:
            r_to_l_disp = feature_based_matching(current_right, current_left, 
                                                patch_size, match_methods[i], 
                                                max_disparities[i])
        else:
            r_to_l_disp = region_based_matching(current_right, current_left, 
                                              template_sizes[i], match_methods[i], 
                                              max_disparities[i], disp_init,
                                              search_ranges[i])
        
        # Perform validity check
        valid_disp = perform_validity_check(l_to_r_disp, r_to_l_disp)
        
        # Fill gaps
        filled_disp = fill_gaps(valid_disp)
        
        # If not at the finest level, upsample for next level
        if i < levels - 1:
            next_h, next_w = left_pyramid[i+1].shape
            disp_init = resize(filled_disp, (next_h, next_w), anti_aliasing=True)
            # Scale disparity values
            disp_init *= 2
        else:
            # At finest level, return filled disparity
            return filled_disp
    
    # Should not reach here
    return None


def save_disparity_map(disparity: np.ndarray, output_dir: str, filename: str) -> None:
    """
    Save disparity map as PNG image.
    
    Args:
        disparity: Disparity map
        output_dir: Output directory
        filename: Filename for the saved image
    """
    # Normalize disparity for visualization
    normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image
    cv2.imwrite(os.path.join(output_dir, filename), normalized)


def generate_disparity_maps(left_img, right_img, levels, template_sizes, match_methods,
                         max_disparities, feature_levels, patch_size, search_ranges):
    """
    Generate both left-to-right and right-to-left disparity maps.
    
    Args:
        left_img: Left image
        right_img: Right image
        levels: Number of pyramid levels
        template_sizes: List of template sizes for each level
        match_methods: List of matching methods for each level
        max_disparities: List of maximum disparities for each level
        feature_levels: List of levels where feature-based matching should be used
        patch_size: Patch size for feature-based matching
        search_ranges: List of search ranges for each level
    
    Returns:
        Tuple of (left-to-right disparity map, right-to-left disparity map)
    """
    # Generate left-to-right disparity map
    print("  Generating left-to-right disparity map...")
    left_to_right = multi_resolution_stereo(
        left_img, right_img, levels, template_sizes, match_methods,
        max_disparities, feature_levels, patch_size, search_ranges
    )
    
    # Generate right-to-left disparity map
    print("  Generating right-to-left disparity map...")
    right_to_left = multi_resolution_stereo(
        right_img, left_img, levels, template_sizes, match_methods,
        max_disparities, feature_levels, patch_size, search_ranges
    )
    
    return left_to_right, right_to_left


def main():
    parser = argparse.ArgumentParser(description='Stereo Analysis System')
    
    # Dataset and output parameters
    parser.add_argument('--image_dir', type=str, default='./images',
                        help='Directory containing stereo datasets')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--stereo_pair', type=str, default='all',
                        choices=['all', 'barn1', 'bull', 'poster', 'sawtooth', 'venus'],
                        help='Stereo pair to process')
    
    # Method parameters
    parser.add_argument('--levels', type=int, default=3,
                        help='Number of pyramid levels')
    parser.add_argument('--match_method', type=str, default='NCC',
                        choices=['SAD', 'SSD', 'NCC'],
                        help='Matching method')
    parser.add_argument('--template_height', type=int, default=7,
                        help='Template height for region-based matching')
    parser.add_argument('--template_width', type=int, default=7,
                        help='Template width for region-based matching')
    parser.add_argument('--max_disparity', type=int, default=64,
                        help='Maximum disparity')
    parser.add_argument('--feature_based_levels', type=str, default='',
                        help='Comma-separated list of levels to use feature-based matching (0-indexed)')
    parser.add_argument('--patch_size', type=int, default=11,
                        help='Patch size for feature-based matching')
    parser.add_argument('--search_range', type=int, default=5,
                        help='Search range around initial disparity')
    
    args = parser.parse_args()
    
    # Process stereo pairs
    stereo_pairs = ['barn1', 'bull', 'poster', 'sawtooth', 'venus'] if args.stereo_pair == 'all' else [args.stereo_pair]
    
    # Parse feature-based levels
    feature_levels = [int(level) for level in args.feature_based_levels.split(',') if level] if args.feature_based_levels else []
    
    for pair in stereo_pairs:
        print(f"Processing stereo pair: {pair}")
        
        # Read images
        left_img, right_img = read_images(args.image_dir, pair)
        
        # Define parameters for each level
        template_sizes = [(args.template_height, args.template_width)] * args.levels
        match_methods = [args.match_method] * args.levels
        max_disparities = [args.max_disparity // (2 ** (args.levels - i - 1)) for i in range(args.levels)]
        search_ranges = [args.search_range] * args.levels
        
        # Generate both disparity maps
        left_to_right, right_to_left = generate_disparity_maps(
            left_img, right_img, args.levels, template_sizes, match_methods,
            max_disparities, feature_levels, args.patch_size, search_ranges
        )
        
        # Create base parameter string
        params_str = f"{pair}_lvl{args.levels}_meth{args.match_method}_temp{args.template_height}x{args.template_width}_maxdisp{args.max_disparity}"
        if feature_levels:
            params_str += f"_feat{'_'.join(map(str, feature_levels))}"
        
        # Save left-to-right disparity map
        save_disparity_map(left_to_right, args.output_dir, f"{params_str}_left_to_right.png")
        
        # Save right-to-left disparity map
        save_disparity_map(right_to_left, args.output_dir, f"{params_str}_right_to_left.png")
        
        print(f"  Saved disparity maps for {pair}")
    
    print("All processing complete!")


if __name__ == "__main__":
    main()