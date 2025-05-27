import cv2 as cv
import numpy as np
import os

# Core image processing functions for convolution, pyramid operations, and image blending

def convolve(I, H):
    """
    Perform convolution of an image with a kernel without using OpenCV's filter2D.
    
    Args:
        I (numpy.ndarray): Input image (2D or 3D array)
        H (numpy.ndarray): 2D convolution kernel
    
    Returns:
        numpy.ndarray: Convolved image clipped to [0, 255] range
    """
    # Flip the kernel for proper convolution (correlation vs convolution)
    H = np.flipud(np.fliplr(H))
    output = np.zeros_like(I, dtype=np.float32)
    # Calculate padding needed for the kernel
    pad_h, pad_w = H.shape[0] // 2, H.shape[1] // 2
    # Pad the input image to handle convolution at edges
    padded_I = np.pad(I, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    
    # Perform convolution by sliding the kernel over the padded image
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            output[i, j] = (padded_I[i:i+H.shape[0], j:j+H.shape[1]] * H[:, :, None]).sum(axis=(0, 1))
    
    return np.clip(output, 0, 255).astype(np.uint8)

def gaussian_kernel(size, sigma):
    """
    Create a 1D Gaussian kernel for image smoothing.
    
    Args:
        size (int): Size of the kernel (should be odd)
        sigma (float): Standard deviation of the Gaussian distribution
    
    Returns:
        numpy.ndarray: 1D normalized Gaussian kernel
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return kernel / np.sum(kernel)

def reduce(I):
    """
    Reduce an image by half using Gaussian filtering and downsampling.
    This implements the REDUCE operation for creating Gaussian pyramids.
    
    Args:
        I (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Reduced image (half size in each dimension)
    """
    kernel = gaussian_kernel(5, 1.0)
    # Apply separable Gaussian filtering (horizontal then vertical)
    I_blurred = convolve(I, kernel[:, None])  # Blur horizontally
    I_blurred = convolve(I_blurred, kernel[None, :])  # Blur vertically
    return I_blurred[::2, ::2]  # Downsample by taking every other pixel

def expand(I):
    """
    Expand an image by doubling its size with interpolation.
    This implements the EXPAND operation for Laplacian pyramid reconstruction.
    
    Args:
        I (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Expanded image (double size in each dimension)
    """
    h, w = I.shape[:2]
    expanded = np.zeros((h * 2, w * 2, I.shape[2]), dtype=np.float32)
    # Place original pixels in even positions
    expanded[::2, ::2] = I
    # Interpolate horizontally
    expanded[::2, 1:-1:2] = 0.5 * (expanded[::2, :-2:2] + expanded[::2, 2::2])
    expanded[::2, -1] = expanded[::2, -2]  # Handle border case
    # Interpolate vertically
    expanded[1:-1:2, :] = 0.5 * (expanded[:-2:2, :] + expanded[2::2, :])
    expanded[-1, :] = expanded[-2, :]  # Handle border case
    
    # Apply Gaussian smoothing to the interpolated image
    kernel = gaussian_kernel(3, 0.5)
    expanded = convolve(expanded, kernel[:, None])
    expanded = convolve(expanded, kernel[None, :])
    
    return np.clip(expanded, 0, 255).astype(np.uint8)

def gaussianPyramid(I, n):
    """
    Construct a Gaussian Pyramid with n levels.
    Each level is a smoothed and downsampled version of the previous level.
    
    Args:
        I (numpy.ndarray): Input image
        n (int): Number of pyramid levels
    
    Returns:
        list: List of n images forming the Gaussian pyramid
    """
    pyramid = [I]
    for _ in range(n-1):
        I = reduce(I)  # Each level is created by reducing the previous level
        pyramid.append(I)
    return pyramid

def laplacianPyramid(I, n):
    """
    Construct a Laplacian Pyramid with n levels.
    Each level contains the high-frequency details lost between adjacent Gaussian pyramid levels.
    
    Args:
        I (numpy.ndarray): Input image
        n (int): Number of pyramid levels
    
    Returns:
        list: List of n images forming the Laplacian pyramid
    """
    I = I.astype(np.float32)
    # First create Gaussian pyramid
    gaussian_pyr = gaussianPyramid(I, n)
    laplacian_pyr = []
    
    # Create Laplacian pyramid levels by subtracting expanded versions
    # of higher pyramid levels from current levels
    for i in range(n-1):
        current_level = gaussian_pyr[i].astype(np.float32)
        next_level = gaussian_pyr[i+1].astype(np.float32)
        # Expand the next level to match current level's size
        expanded = expand(next_level)
        expanded = cv.resize(expanded, (current_level.shape[1], current_level.shape[0]))
        expanded = expanded.astype(np.float32)
        # Laplacian is the difference between current level and expanded next level
        laplacian = cv.subtract(current_level, expanded)
        laplacian_pyr.append(laplacian)
    
    # The last level is the same as the last Gaussian pyramid level
    laplacian_pyr.append(gaussian_pyr[-1].astype(np.float32))
    return laplacian_pyr

def reconstruct(LI, n):
    """
    Reconstruct an image from its Laplacian Pyramid.
    
    Args:
        LI (list): Laplacian pyramid (list of n images)
        n (int): Number of pyramid levels
    
    Returns:
        numpy.ndarray: Reconstructed image
    """
    # Start with the smallest level
    I_reconstructed = LI[-1].astype(np.float32)
    
    # Iteratively expand and add Laplacian levels
    for i in range(n-2, -1, -1):
        # Expand the current reconstruction
        expanded = expand(I_reconstructed)
        expanded = cv.resize(expanded, (LI[i].shape[1], LI[i].shape[0]))
        expanded = expanded.astype(np.float32)
        # Add the Laplacian level to get the next level of detail
        current_level = LI[i].astype(np.float32)
        I_reconstructed = cv.add(expanded, current_level)
    
    return np.clip(I_reconstructed, 0, 255).astype(np.uint8)

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for selecting points in images.
    Handles point selection, visualization, and coordinate display.
    
    Args:
        event: Mouse event type
        x, y: Mouse coordinates
        flags: Additional flags
        param: Dictionary containing image and points data
    """
    if event == cv.EVENT_LBUTTONDOWN:
        param['points'].append((x, y))
        # Create a copy of the image to avoid modifying the original
        display_img = param['image'].copy()
        
        # Draw all points with numbers for better visualization
        for i, (px, py) in enumerate(param['points']):
            # Draw point with number
            cv.circle(display_img, (px, py), 5, (0, 255, 0), -1)
            cv.putText(display_img, str(i+1), (px+5, py-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update the display image
        param['display_image'] = display_img
        cv.imshow(param['window_name'], display_img)
        print(f"Selected point {len(param['points'])}: ({x}, {y})")
    
    # Show live coordinates on hover for better point selection
    elif event == cv.EVENT_MOUSEMOVE:
        if 'display_image' in param:
            hover_img = param['display_image'].copy()
            cv.putText(hover_img, f"({x}, {y})", (x+10, y), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.imshow(param['window_name'], hover_img)

def select_points(img1, img2, num_points, window_name1="Left Image", window_name2="Right Image"):
    """
    Let user select corresponding points in two images for image alignment.
    
    Args:
        img1 (numpy.ndarray): First image (left)
        img2 (numpy.ndarray): Second image (right)
        num_points (int): Number of point pairs to select
        window_name1 (str): Window name for first image
        window_name2 (str): Window name for second image
    
    Returns:
        tuple: Two numpy arrays containing corresponding points (None if cancelled)
    """
    points1 = []
    points2 = []
    
    # Create windows with proper size for better visualization
    cv.namedWindow(window_name1, cv.WINDOW_NORMAL)
    cv.namedWindow(window_name2, cv.WINDOW_NORMAL)
    
    # Set reasonable window sizes based on image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    cv.resizeWindow(window_name1, min(w1, 800), min(h1, 600))
    cv.resizeWindow(window_name2, min(w2, 800), min(h2, 600))
    
    # Prepare parameter dictionaries for mouse callback
    param1 = {'points': points1, 'image': img1.copy(), 'window_name': window_name1}
    param2 = {'points': points2, 'image': img2.copy(), 'window_name': window_name2}
    
    # Set up mouse callbacks for both windows
    cv.setMouseCallback(window_name1, mouse_callback, param1)
    cv.setMouseCallback(window_name2, mouse_callback, param2)
    
    # Display instructions to user
    print(f"\nPlease select {num_points} corresponding points in each image")
    print("First click in the left image, then the corresponding point in the right image")
    print("Press 'c' to clear all points and start over")
    print("Press 'r' to remove the last point")
    print("Press 'Enter' when done selecting points")
    
    # Show initial images
    cv.imshow(window_name1, img1)
    cv.imshow(window_name2, img2)
    
    while True:
        key = cv.waitKey(1) & 0xFF
        
        # Clear points
        if key == ord('c'):
            points1.clear()
            points2.clear()
            param1['display_image'] = img1.copy()
            param2['display_image'] = img2.copy()
            cv.imshow(window_name1, img1)
            cv.imshow(window_name2, img2)
            print("Points cleared. Please start over.")
        
        # Remove last point
        elif key == ord('r'):
            if points1:
                points1.pop()
                # Redraw remaining points on first image
                display_img1 = img1.copy()
                for i, (px, py) in enumerate(points1):
                    cv.circle(display_img1, (px, py), 5, (0, 255, 0), -1)
                    cv.putText(display_img1, str(i+1), (px+5, py-5), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                param1['display_image'] = display_img1
                cv.imshow(window_name1, display_img1)
            
            if points2:
                points2.pop()
                # Redraw remaining points on second image
                display_img2 = img2.copy()
                for i, (px, py) in enumerate(points2):
                    cv.circle(display_img2, (px, py), 5, (0, 255, 0), -1)
                    cv.putText(display_img2, str(i+1), (px+5, py-5), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                param2['display_image'] = display_img2
                cv.imshow(window_name2, display_img2)
            
            print("Removed last point.")
        
        # Confirm selection
        elif key == 13:  # Enter key
            if len(points1) >= num_points and len(points2) >= num_points:
                break
            else:
                print(f"Please select at least {num_points} points in each image")
        
        # Quit
        elif key == 27:  # Escape key
            print("Selection cancelled.")
            cv.destroyAllWindows()
            return None, None
    
    cv.destroyAllWindows()
    return np.float32(points1[:num_points]), np.float32(points2[:num_points])

# --- Functions for manual blend-boundary selection ---

def select_blend_boundary(image, window_name="Select Blend Boundary"):
    """
    Allow user to select blend boundary points on the image for creating a blending mask.
    
    Args:
        image (numpy.ndarray): Image to select blend boundary on
        window_name (str): Name of the display window
    
    Returns:
        numpy.ndarray: Array of selected boundary points (empty if cancelled)
    """
    blend_points = []
    temp_image = image.copy()
    display_image = image.copy()
    
    def blend_mouse_callback(event, x, y, flags, param):
        nonlocal display_image
        
        if event == cv.EVENT_LBUTTONDOWN:
            param.append((x, y))
            # Update display image with current points and lines
            display_image = temp_image.copy()
            
            # Draw connecting lines between points
            if len(param) > 1:
                points_array = np.array(param, dtype=np.int32)
                cv.polylines(display_image, [points_array], False, (0, 255, 0), 2)
            
            # Draw individual points with numbers
            for i, (px, py) in enumerate(param):
                cv.circle(display_image, (px, py), 5, (255, 0, 0), -1)
                cv.putText(display_image, str(i+1), (px+5, py-5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv.imshow(window_name, display_image)
            print(f"Blend boundary point {len(param)}: ({x}, {y})")
        
        elif event == cv.EVENT_MOUSEMOVE:
            # Show preview line from last point to current mouse position
            if len(param) > 0:
                preview = display_image.copy()
                last_point = param[-1]
                cv.line(preview, last_point, (x, y), (0, 255, 255), 1)
                cv.imshow(window_name, preview)
    
    # Create window with proper size
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    h, w = image.shape[:2]
    cv.resizeWindow(window_name, min(w, 800), min(h, 600))
    
    cv.imshow(window_name, temp_image)
    cv.setMouseCallback(window_name, lambda event, x, y, flags, param: blend_mouse_callback(event, x, y, flags, blend_points))
    
    print("Please select blend boundary points (ideally a curve from top to bottom)")
    print("Press 'c' to clear all points and start over")
    print("Press 'r' to remove the last point")
    print("Press 'Enter' when done to confirm selection")
    print("Press 'Esc' to cancel")
    
    while True:
        key = cv.waitKey(1) & 0xFF
        
        # Clear points
        if key == ord('c'):
            blend_points.clear()
            display_image = temp_image.copy()
            cv.imshow(window_name, display_image)
            print("Points cleared. Please start over.")
        
        # Remove last point
        elif key == ord('r'):
            if blend_points:
                blend_points.pop()
                # Redraw remaining points
                display_image = temp_image.copy()
                if len(blend_points) > 1:
                    points_array = np.array(blend_points, dtype=np.int32)
                    cv.polylines(display_image, [points_array], False, (0, 255, 0), 2)
                
                # Draw individual points
                for i, (px, py) in enumerate(blend_points):
                    cv.circle(display_image, (px, py), 5, (255, 0, 0), -1)
                    cv.putText(display_image, str(i+1), (px+5, py-5), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.imshow(window_name, display_image)
                print("Removed last point.")
        
        # Confirm selection
        elif key == 13:  # Enter key
            break
        
        # Cancel
        elif key == 27:  # Escape key
            print("Selection cancelled.")
            cv.destroyWindow(window_name)
            return np.array([], dtype=np.int32)
    
    cv.destroyWindow(window_name)
    return np.array(blend_points, dtype=np.int32)

def create_mask_from_boundary(image_shape, boundary_points, blur_radius=21):
    """
    Create a blending mask from user-selected boundary points.
    
    Args:
        image_shape (tuple): Shape of the target image (height, width)
        boundary_points (numpy.ndarray): Array of boundary points
        blur_radius (int): Radius for Gaussian blur of the mask edges
    
    Returns:
        numpy.ndarray: 3-channel blending mask with smooth transitions
    """
    h, w = image_shape[:2]
    
    # Create default mask if no points were selected
    if boundary_points.shape[0] < 2:
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:, :w//2] = 1.0  # Default vertical split
        mask = cv.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        return cv.merge([mask, mask, mask])
    
    # Sort points by y-coordinate for consistent mask creation
    sorted_points = boundary_points[boundary_points[:,1].argsort()]
    
    # Add top and bottom points if needed to complete the boundary
    if sorted_points[0,1] > 0:
        x_top = sorted_points[0,0]
        sorted_points = np.vstack(([[x_top, 0]], sorted_points))
    
    if sorted_points[-1,1] < h-1:
        x_bottom = sorted_points[-1,0]
        sorted_points = np.vstack((sorted_points, [[x_bottom, h-1]]))
    
    # Create closed polygon for the left side of the blend
    poly_left = np.vstack(([[0,0]], [[0,h-1]], sorted_points[::-1]))
    
    # Create binary mask
    mask = np.zeros((h, w), dtype=np.float32)
    cv.fillPoly(mask, [poly_left.astype(np.int32)], 1)
    
    # Apply Gaussian blur for smoother transition
    mask = cv.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    
    # Normalize mask values to [0,1] range
    if mask.max() > 0:
        mask = mask / mask.max()
    
    # Create 3-channel mask
    mask = cv.merge([mask, mask, mask])
    
    return mask

def blend_images(img1, img2, mask):
    """
    Blend two images using Laplacian pyramid blending technique.
    
    Args:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
        mask (numpy.ndarray): Blending mask (3-channel, values in [0,1])
    
    Returns:
        numpy.ndarray: Blended image
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mask = mask.astype(np.float32)
    
    # Ensure all images have the same size
    h, w = img1.shape[:2]
    img2 = cv.resize(img2, (w, h))
    mask = cv.resize(mask, (w, h))
    
    # Create Laplacian pyramids for both images and Gaussian pyramid for mask
    levels = 4  # Number of pyramid levels
    pyr1 = laplacianPyramid(img1, levels)
    pyr2 = laplacianPyramid(img2, levels)
    mask_pyr = gaussianPyramid(mask, levels)
    
    # Blend each level of the pyramid
    blended_pyr = []
    for i in range(levels):
        # Resize mask to match current pyramid level
        mask_resized = cv.resize(mask_pyr[i], 
                               (pyr1[i].shape[1], pyr1[i].shape[0])).astype(np.float32)
        level1 = pyr1[i].astype(np.float32)
        level2 = pyr2[i].astype(np.float32)
        # Blend current level using the mask
        blended = level1 * mask_resized + level2 * (1.0 - mask_resized)
        blended_pyr.append(blended)
    
    # Reconstruct the final image from the blended pyramid
    return reconstruct(blended_pyr, levels)

# --- Bounding Box Shift Function ---
def bounding_box_shift(img):
    """
    Shift the mosaic image based on the bounding box of non-empty regions.
    Removes unnecessary black borders from the image.
    
    Args:
        img (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Cropped image without black borders
    """
    gray = cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2GRAY)
    coords = cv.findNonZero(gray)
    if coords is None:
        return img
    x, y, w, h = cv.boundingRect(coords)
    shifted = img[y:y+h, x:x+w]
    return shifted

def warp_image(img, points_src, points_dst, method='affine'):
    """
    Compute transformation matrix and warp image corners.
    
    Args:
        img (numpy.ndarray): Input image to be warped
        points_src (numpy.ndarray): Source points for transformation
        points_dst (numpy.ndarray): Destination points for transformation
        method (str): Transformation method ('affine' or 'perspective')
    
    Returns:
        tuple: (transformation_matrix, warped_corners)
    """
    # Get image corners for computing the complete warped image bounds
    h, w = img.shape[:2]
    corners = np.array([
        [0,0], [w,0], [w,h], [0,h]
    ], dtype=np.float32).reshape(-1,1,2)

    if method == 'affine':
        # Compute affine transformation (2x3 matrix)
        M_affine = cv.getAffineTransform(points_src[:3], points_dst[:3])
        # Convert to full 3x3 matrix for consistency
        M_3x3 = np.eye(3, dtype=np.float32)
        M_3x3[:2] = M_affine
        # Transform corners using affine matrix
        warped_corners = cv.transform(corners, M_affine)
        return M_3x3, warped_corners
    else:
        # Compute perspective transformation (3x3 matrix)
        M = cv.getPerspectiveTransform(points_src, points_dst)
        # Transform corners using perspective matrix
        warped_corners = cv.perspectiveTransform(corners, M)
        return M, warped_corners

def mosaic_images(images, method='affine'):
    """Create image mosaic using specified warping method.
    
    If method=='none', corresponding points are still selected and used to compute a
    simple translation offset to align the images (no full warp is applied).
    """
    if len(images) < 2:
        return images[0]
    
    # Start with the first image as the current mosaic result.
    result = images[0].astype(np.float32)
    
    for i in range(1, len(images)):
        print(f"\nProcessing image {i+1}...")

        if method == None:
            # ---- No full warp; use selected corresponding points to compute a translation offset ----
            print("Please select corresponding points between the current mosaic and the new image for translation alignment.")
            # Here, you can choose one pair or more for averaging.
            num_points = 3  # For instance, use 3 pairs to average out noise.
            pts_result, pts_new = select_points(result.astype(np.uint8),
                                                images[i],
                                                num_points,
                                                f"Current Mosaic",
                                                f"New Image {i}")
            # Compute average translation offset (from new image to current mosaic)
            offset = np.mean(pts_result - pts_new, axis=0)
            print("Computed translation offset:", offset)
            # Build a translation matrix T based on the computed offset.
            T = np.array([[1, 0, offset[0]],
                          [0, 1, offset[1]],
                          [0, 0, 1]], dtype=np.float32)
            
            # Compute bounding box for the current mosaic and the new image translated by T.
            h_result, w_result = result.shape[:2]
            h_new, w_new = images[i].shape[:2]
            corners_result = np.array([[0, 0],
                                       [w_result, 0],
                                       [w_result, h_result],
                                       [0, h_result]], dtype=np.float32)
            # Compute corners for the new image after translation.
            corners_new = np.array([[0, 0],
                                      [w_new, 0],
                                      [w_new, h_new],
                                      [0, h_new]], dtype=np.float32)
            # Convert corners_new to homogeneous and apply T.
            corners_new_h = np.hstack([corners_new, np.ones((4,1))]).T  # shape (3,4)
            warped_new = T @ corners_new_h
            warped_new = (warped_new / warped_new[2, :]).T[:, :2]
            
            # Combine corners from result and translated new image.
            all_corners = np.vstack([corners_result, warped_new])
            min_xy = np.floor(all_corners.min(axis=0)).astype(int)
            max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
            shift = -np.minimum(min_xy, 0)
            canvas_width  = max_xy[0] + shift[0]
            canvas_height = max_xy[1] + shift[1]
            
            # Build a translation matrix for the overall canvas shift.
            T_shift = np.array([[1, 0, shift[0]],
                                [0, 1, shift[1]],
                                [0, 0, 1]], dtype=np.float32)
            
            # Warp new image: here, since T is a pure translation, we can use warpAffine.
            canvas_new = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            # Compose overall transform: first T then T_shift.
            overall_T = T_shift @ T
            cv.warpAffine(images[i].astype(np.float32),
                          overall_T[:2],
                          (canvas_width, canvas_height),
                          dst=canvas_new,
                          borderMode=cv.BORDER_TRANSPARENT)
            
            # Warp the current mosaic with T_shift.
            canvas_result = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            cv.warpAffine(result,
                          T_shift[:2],
                          (canvas_width, canvas_height),
                          dst=canvas_result,
                          borderMode=cv.BORDER_TRANSPARENT)
            
            # Create alpha masks based on where images exist (non-zero)
            # This ensures we don't get black areas in the final result
            alpha_result = (np.sum(canvas_result, axis=2) > 0).astype(np.float32)
            alpha_new = (np.sum(canvas_new, axis=2) > 0).astype(np.float32)
            
            # Expand dimensions for broadcasting
            alpha_result = np.expand_dims(alpha_result, axis=2)
            alpha_new = np.expand_dims(alpha_new, axis=2)
            
            # Detect overlapping regions
            overlap = alpha_result * alpha_new
            
            # Create a gradient blend in overlapping regions
            # We'll use a distance-based approach for better blending
            if np.sum(overlap) > 0:
                # Create a linear mask for overlapping regions
                mask = create_linear_mask((canvas_height, canvas_width))
                
                # Apply the mask only to overlapping regions
                blended = np.zeros_like(canvas_result)
                
                # Where only result exists, keep result
                result_only = (alpha_result > 0) & (alpha_new == 0)
                blended += canvas_result * result_only
                
                # Where only new image exists, keep new image
                new_only = (alpha_new > 0) & (alpha_result == 0)
                blended += canvas_new * new_only
                
                # Where both exist, blend according to mask
                both = (alpha_result > 0) & (alpha_new > 0)
                both_expanded = np.repeat(both, 3, axis=2)
                blended += both_expanded * (canvas_result * mask + canvas_new * (1 - mask))
            else:
                # If there's no overlap, simply combine them
                blended = canvas_result + canvas_new
            
            # Optionally, trim empty borders.
            blended = bounding_box_shift(blended)
            
            result = blended.astype(np.float32)
        
        else:
            # ---- Affine or perspective branch (existing code) ----
            print("Please select corresponding points between the current mosaic and the new image for unwarping.")
            num_points = 4 if method == 'perspective' else 3
            points1, points2 = select_points(result.astype(np.uint8),
                                             images[i],
                                             num_points,
                                             f"Result {i-1}",
                                             f"Image {i}")
            
            # 1) Compute transformation and get warped corners of the new image.
            M, warped_corners = warp_image(images[i], points2, points1, method)
            
            # 2) Compute corners of the current mosaic.
            hR, wR = result.shape[:2]
            corners_result = np.array([[0, 0],
                                       [wR, 0],
                                       [wR, hR],
                                       [0, hR]], dtype=np.float32).reshape(-1, 1, 2)
            
            # 3) Combine corners to determine the bounding box.
            all_corners = np.vstack((corners_result, warped_corners)).reshape(-1, 2)
            min_x, min_y = np.floor(all_corners.min(axis=0))
            max_x, max_y = np.ceil(all_corners.max(axis=0))
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)
            
            # 4) Compute translation to make coordinates positive.
            shift_x = -min_x if min_x < 0 else 0
            shift_y = -min_y if min_y < 0 else 0
            
            # 5) Final canvas size.
            canvas_width  = max_x + shift_x
            canvas_height = max_y + shift_y
            
            # 6) Build translation matrix T.
            T = np.array([[1, 0, shift_x],
                          [0, 1, shift_y],
                          [0, 0, 1]], dtype=np.float32)
            
            # 7) Combine T with M for warping the new image.
            M_shifted = T @ M
            
            # 8) Warp the new image onto the canvas.
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            warped_img2 = cv.warpPerspective(images[i].astype(np.float32),
                                             M_shifted,
                                             (canvas_width, canvas_height),
                                             dst=canvas,
                                             borderMode=cv.BORDER_TRANSPARENT)
            
            # 9) Shift the current mosaic into the same canvas.
            canvas_result = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            if method == 'affine':
                T_affine = T[:2]
                cv.warpAffine(result,
                              T_affine,
                              (canvas_width, canvas_height),
                              dst=canvas_result,
                              borderMode=cv.BORDER_TRANSPARENT)
            else:
                cv.warpPerspective(result,
                                   T,
                                   (canvas_width, canvas_height),
                                   dst=canvas_result,
                                   borderMode=cv.BORDER_TRANSPARENT)
            
            # 10) Blend overlapping regions.
            composite = cv.addWeighted(canvas_result.astype(np.uint8), 0.5,
                                       canvas.astype(np.uint8), 0.5, 0)
            print("Please select the blend boundary for the overlapping region.")
            boundary_points = select_blend_boundary(composite.copy(), "Select Blend Boundary for Mosaic")
            mask = create_mask_from_boundary((canvas_height, canvas_width), boundary_points)
            blended = blend_images(canvas_result, canvas, mask)
            
            # 11) Optionally, trim empty borders.
            blended = bounding_box_shift(blended)
            result = blended.astype(np.float32)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_linear_mask(image_shape):
    """
    Create a linear blending mask transitioning from left to right.
    
    Args:
        image_shape (tuple): Shape of the target image (height, width)
    
    Returns:
        numpy.ndarray: 3-channel mask with linear gradient from 1 (left) to 0 (right)
    """
    h, w = image_shape[:2]
    # Create horizontal linear gradient
    mask = np.tile(np.linspace(1, 0, w, dtype=np.float32), (h, 1))
    # Convert to 3-channel mask
    mask = cv.merge([mask, mask, mask])
    return mask
    

def automatic_mosaic(img1, img2, patch_size=11, ncc_threshold=0.7):
    """
    Automatically compute point correspondences between images and create a mosaic.
    Uses Harris corner detection and normalized cross-correlation for feature matching.
    
    Args:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
        patch_size (int): Size of patches for correlation comparison
        ncc_threshold (float): Threshold for normalized cross-correlation (0 to 1)
    
    Returns:
        tuple: (mosaic_image, (matched_points1, matched_points2))
    """
    # Convert images to grayscale for feature detection
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    # Detect corners using Harris detector via goodFeaturesToTrack
    maxCorners = 1000  # Maximum number of corners to detect
    qualityLevel = 0.01  # Quality level for corner detection
    minDistance = 10  # Minimum distance between corners
    blockSize = 3  # Size of window for corner detection
    useHarrisDetector = True
    k = 0.04  # Harris detector free parameter

    # Detect corners in both images
    corners1 = cv.goodFeaturesToTrack(gray1, maxCorners, qualityLevel, minDistance,
                                      blockSize=blockSize, useHarrisDetector=useHarrisDetector, k=k)
    corners2 = cv.goodFeaturesToTrack(gray2, maxCorners, qualityLevel, minDistance,
                                      blockSize=blockSize, useHarrisDetector=useHarrisDetector, k=k)
    
    if corners1 is None or corners2 is None:
        print("No corners detected.")
        return None, None

    # Convert corners to integer coordinates
    corners1 = np.int0(corners1).reshape(-1, 2)
    corners2 = np.int0(corners2).reshape(-1, 2)
    
    half_patch = patch_size // 2
    pts1 = []
    pts2 = []
    
    # For each corner in img1, find the best matching corner in img2 using NCC
    for pt1 in corners1:
        x1, y1 = pt1
        # Ensure the patch from img1 is fully within the image boundaries
        if (x1 - half_patch < 0 or x1 + half_patch >= gray1.shape[1] or 
            y1 - half_patch < 0 or y1 + half_patch >= gray1.shape[0]):
            continue
        
        # Extract patch from first image
        patch1 = gray1[y1-half_patch:y1+half_patch+1, x1-half_patch:x1+half_patch+1]
        best_ncc = -1.0
        best_pt2 = None
        
        # Find best matching patch in second image
        for pt2 in corners2:
            x2, y2 = pt2
            # Ensure the patch from img2 is fully within the image boundaries
            if (x2 - half_patch < 0 or x2 + half_patch >= gray2.shape[1] or 
                y2 - half_patch < 0 or y2 + half_patch >= gray2.shape[0]):
                continue
            
            # Extract patch from second image
            patch2 = gray2[y2-half_patch:y2+half_patch+1, x2-half_patch:x2+half_patch+1]
            
            # Compute normalized cross-correlation (NCC)
            patch1_mean = np.mean(patch1)
            patch2_mean = np.mean(patch2)
            patch1_std = np.std(patch1)
            patch2_std = np.std(patch2)
            
            # Skip patches with no variation
            if patch1_std < 1e-6 or patch2_std < 1e-6:
                continue
                
            # Calculate NCC score
            ncc = np.sum((patch1 - patch1_mean) * (patch2 - patch2_mean)) / (patch_size * patch_size * patch1_std * patch2_std)
            
            # Update best match if current NCC is higher
            if ncc > best_ncc:
                best_ncc = ncc
                best_pt2 = pt2
        
        # Add point pair if NCC score is above threshold
        if best_ncc >= ncc_threshold and best_pt2 is not None:
            pts1.append([x1, y1])
            pts2.append([best_pt2[0], best_pt2[1]])
    
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    if pts1.shape[0] < 4:
        print("Not enough good matches found.")
        return None, None
    
    # Compute homography with RANSAC for robust estimation
    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)
    
    if H is None:
        print("Could not compute homography.")
        return None, None
    
    # Calculate dimensions of output mosaic
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Get corners of img2 and transform them using the computed homography
    corners2_rect = np.array([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners2 = cv.perspectiveTransform(corners2_rect, H)
    
    # Get corners of first image
    corners1_rect = np.array([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Combine all corners to determine output size
    all_corners = np.vstack((corners1_rect.reshape(-1, 2), warped_corners2.reshape(-1, 2)))
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    
    # Calculate offset to ensure all points are positive
    offset_x = max(0, -min_x)
    offset_y = max(0, -min_y)
    
    output_width = max_x + offset_x
    output_height = max_y + offset_y
    
    # Create translation matrix and combine with homography
    translation_matrix = np.array([[1, 0, offset_x],
                                   [0, 1, offset_y],
                                   [0, 0, 1]], dtype=np.float32)
    warped_H = translation_matrix @ H
    
    # Warp and blend images
    warped_img2 = cv.warpPerspective(img2, warped_H, (output_width, output_height))
    img1_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
    y_end = min(offset_y + h1, output_height)
    x_end = min(offset_x + w1, output_width)
    img1_canvas[offset_y:y_end, offset_x:x_end] = img1[:y_end-offset_y, :x_end-offset_x]
    
    # Create masks for blending
    img1_mask = np.sum(img1_canvas, axis=2) > 0
    img2_mask = np.sum(warped_img2, axis=2) > 0
    overlap = img1_mask & img2_mask
    
    # Initialize result image
    result = np.zeros_like(img1_canvas)
    
    # Copy non-overlapping regions directly
    result[img1_mask & ~overlap] = img1_canvas[img1_mask & ~overlap]
    result[img2_mask & ~overlap] = warped_img2[img2_mask & ~overlap]
    
    # Blend overlapping regions using distance-based weights
    if np.any(overlap):
        # Compute distance transforms for weight calculation
        dist1 = cv.distanceTransform((~img2_mask).astype(np.uint8), cv.DIST_L2, 3)
        dist2 = cv.distanceTransform((~img1_mask).astype(np.uint8), cv.DIST_L2, 3)
        weight = dist1 / (dist1 + dist2 + 1e-6)
        # Smooth weights for better blending
        weight = cv.GaussianBlur(weight, (15, 15), 0)
        
        # Blend each color channel
        for c in range(3):
            result[:,:,c][overlap] = (
                weight[overlap] * img1_canvas[:,:,c][overlap] +
                (1 - weight[overlap]) * warped_img2[:,:,c][overlap]
            )
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Crop the mosaic to remove unnecessary black borders
    gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_result, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        all_contours = np.vstack(contours)
        x, y, w, h = cv.boundingRect(all_contours)
        result = result[y:y+h, x:x+w]
    
    return result, (pts1, pts2)

def generate_outputs():
    """
    Generate and save all required output images for the assignment.
    This includes basic operations (convolution, reduce, expand),
    pyramid operations, and image mosaics using different methods.
    """
    # Create output directory
    output_dir = "PartB/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define image pairs for mosaicking
    image_pairs = [
        ('PartB/image1_left.png', 'PartB/image1_right.png', 'affine'),
        ('PartB/image2_left.png', 'PartB/image2_right.png', 'perspective'),
        ('PartB/image3_left.png', 'PartB/image3_right.png', 'affine'),
        ('PartB/image4_left.png', 'PartB/image4_right.png', 'perspective')
    ]
    
    # Load test image
    img = cv.imread('PartB/lena.png')
    if img is None:
        raise FileNotFoundError("Image not found: PartB/lena.png")
    
    # Generate basic operation outputs
    cv.imwrite(f'{output_dir}/convolved.png', 
               convolve(img, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])))  # example kernel
    cv.imwrite(f'{output_dir}/reduced.png', reduce(img))
    cv.imwrite(f'{output_dir}/expanded.png', expand(img))
    
    # Generate and save Gaussian pyramid levels
    gaussian_pyr = gaussianPyramid(img, 3)
    for i, level in enumerate(gaussian_pyr):
        cv.imwrite(f'{output_dir}/gaussian_pyramid_level_{i}.png', level)
    
    # Generate and save Laplacian pyramid levels
    laplacian_pyr = laplacianPyramid(img, 3)
    for i, level in enumerate(laplacian_pyr):
        cv.imwrite(f'{output_dir}/laplacian_pyramid_level_{i}.png', level)
    
    # Test and evaluate reconstruction
    print("Testing reconstruction...")
    reconstructed = reconstruct(laplacianPyramid(img, 3), 3)
    cv.imwrite(f'{output_dir}/reconstructed.png', reconstructed)
    
    # Calculate and save reconstruction error metrics
    mse = np.mean((img.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    print(f"Reconstruction MSE: {mse:.4f}")
    
    error_file = os.path.join(output_dir, 'reconstruction_error.txt')
    with open(error_file, 'w') as f:
        f.write(f"Reconstruction Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Reconstruction Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}\n")
    print(f"Reconstruction error saved to: {error_file}")
    
    # Generate image mosaics using different methods
    print("\nStarting image mosaicking (manual correspondences)...")
    for i, (left_path, right_path, preferred_method) in enumerate(image_pairs, 1):
        print(f"\nProcessing image pair {i}...")
        img1 = cv.imread(left_path)
        img2 = cv.imread(right_path)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not load images from {left_path} and {right_path}")
            continue
        
        # Create mosaic with specified transformation method
        print(f"\nCreating mosaic with {preferred_method} transformation...")
        mosaic_warped = mosaic_images([img1, img2], method=preferred_method)
        cv.imwrite(f'{output_dir}/mosaic_pair{i}_{preferred_method}.png', mosaic_warped)
        
        # Create mosaic without transformation (translation only)
        print("\nCreating mosaic without transformation...")
        mosaic_none = mosaic_images([img1, img2], method=None)
        cv.imwrite(f'{output_dir}/mosaic_pair{i}_none.png', mosaic_none)
    
    print("\nMosaic generation complete!")
    print("Generated 8 mosaicked images using manual correspondences.")

def generate_auto_mosaics():
    """
    Generate mosaicked images using automatic feature detection and matching.
    This function processes all image pairs using the automatic_mosaic function.
    """
    # Create output directory for automatic mosaics
    output_dir = "PartB/output_auto"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define image pairs to process
    image_pairs = [
        ('PartB/image1_left.png', 'PartB/image1_right.png'),
        ('PartB/image2_left.png', 'PartB/image2_right.png'),
        ('PartB/image3_left.png', 'PartB/image3_right.png'),
        ('PartB/image4_left.png', 'PartB/image4_right.png')
    ]
    
    # Process each image pair
    for i, (img1_path, img2_path) in enumerate(image_pairs, 1):
        print(f"\nProcessing automatic mosaic for image pair {i}...")
        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not load images from {img1_path} and {img2_path}")
            continue
        
        # Generate automatic mosaic
        mosaic, matches = automatic_mosaic(img1, img2)
        if mosaic is not None:
            cv.imwrite(f'{output_dir}/auto_mosaic_pair{i}.png', mosaic)
            print(f"Automatic mosaic for pair {i} saved.")
        else:
            print(f"Automatic mosaic for pair {i} could not be generated due to insufficient correspondences.")

if __name__ == "__main__":
    # Generate all outputs including manual and automatic mosaics
    generate_outputs()
    generate_auto_mosaics()
