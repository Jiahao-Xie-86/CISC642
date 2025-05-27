import cv2 as cv
import numpy as np
import os

# Core image processing functions for convolution, pyramid operations, and image blending

def convolve(I, H):
    """Perform convolution of an image with a kernel without using OpenCV's filter2D."""
    # Flip the kernel for proper convolution
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
    """Create a 1D Gaussian kernel for image smoothing."""
    ax = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return kernel / np.sum(kernel)

def reduce(I):
    """Reduce an image by half using Gaussian filtering and downsampling."""
    kernel = gaussian_kernel(5, 1.0)
    I_blurred = convolve(I, kernel[:, None])  # Blur horizontally
    I_blurred = convolve(I_blurred, kernel[None, :])  # Blur vertically
    return I_blurred[::2, ::2]

def expand(I):
    """Expand an image by doubling its size with interpolation."""
    h, w = I.shape[:2]
    expanded = np.zeros((h * 2, w * 2, I.shape[2]), dtype=np.float32)
    expanded[::2, ::2] = I
    expanded[::2, 1:-1:2] = 0.5 * (expanded[::2, :-2:2] + expanded[::2, 2::2])
    expanded[::2, -1] = expanded[::2, -2]
    expanded[1:-1:2, :] = 0.5 * (expanded[:-2:2, :] + expanded[2::2, :])
    expanded[-1, :] = expanded[-2, :]
    
    kernel = gaussian_kernel(3, 0.5)
    expanded = convolve(expanded, kernel[:, None])
    expanded = convolve(expanded, kernel[None, :])
    
    return np.clip(expanded, 0, 255).astype(np.uint8)

def gaussianPyramid(I, n):
    """Construct a Gaussian Pyramid with n levels."""
    pyramid = [I]
    for _ in range(n-1):
        I = reduce(I)
        pyramid.append(I)
    return pyramid

def laplacianPyramid(I, n):
    """Construct a Laplacian Pyramid with n levels."""
    I = I.astype(np.float32)
    gaussian_pyr = gaussianPyramid(I, n)
    laplacian_pyr = []
    
    for i in range(n-1):
        current_level = gaussian_pyr[i].astype(np.float32)
        next_level = gaussian_pyr[i+1].astype(np.float32)
        expanded = expand(next_level)
        expanded = cv.resize(expanded, (current_level.shape[1], current_level.shape[0]))
        expanded = expanded.astype(np.float32)
        laplacian = cv.subtract(current_level, expanded)
        laplacian_pyr.append(laplacian)
    
    laplacian_pyr.append(gaussian_pyr[-1].astype(np.float32))
    return laplacian_pyr

def reconstruct(LI, n):
    """Reconstruct an image from its Laplacian Pyramid."""
    I_reconstructed = LI[-1].astype(np.float32)
    for i in range(n-2, -1, -1):
        expanded = expand(I_reconstructed)
        expanded = cv.resize(expanded, (LI[i].shape[1], LI[i].shape[0]))
        expanded = expanded.astype(np.float32)
        current_level = LI[i].astype(np.float32)
        I_reconstructed = cv.add(expanded, current_level)
    return np.clip(I_reconstructed, 0, 255).astype(np.uint8)

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for selecting points in images."""
    if event == cv.EVENT_LBUTTONDOWN:
        param['points'].append((x, y))
        # Create a copy of the image to avoid modifying the original
        display_img = param['image'].copy()
        
        # Draw all points with numbers
        for i, (px, py) in enumerate(param['points']):
            # Draw point with number
            cv.circle(display_img, (px, py), 5, (0, 255, 0), -1)
            cv.putText(display_img, str(i+1), (px+5, py-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update the display image
        param['display_image'] = display_img
        cv.imshow(param['window_name'], display_img)
        
        print(f"Selected point {len(param['points'])}: ({x}, {y})")
    
    # Show point coordinates on hover for better positioning
    elif event == cv.EVENT_MOUSEMOVE:
        if 'display_image' in param:
            hover_img = param['display_image'].copy()
            cv.putText(hover_img, f"({x}, {y})", (x+10, y), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.imshow(param['window_name'], hover_img)

def select_points(img1, img2, num_points, window_name1="Left Image", window_name2="Right Image"):
    """Let user select corresponding points in two images."""
    points1 = []
    points2 = []
    
    # Create windows with proper size
    cv.namedWindow(window_name1, cv.WINDOW_NORMAL)
    cv.namedWindow(window_name2, cv.WINDOW_NORMAL)
    
    # Set window sizes
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    cv.resizeWindow(window_name1, min(w1, 800), min(h1, 600))
    cv.resizeWindow(window_name2, min(w2, 800), min(h2, 600))
    
    # Prepare original images
    param1 = {'points': points1, 'image': img1.copy(), 'window_name': window_name1}
    param2 = {'points': points2, 'image': img2.copy(), 'window_name': window_name2}
    
    # Set up mouse callbacks
    cv.setMouseCallback(window_name1, mouse_callback, param1)
    cv.setMouseCallback(window_name2, mouse_callback, param2)
    
    # Instructions
    print(f"\nPlease select {num_points} corresponding points in each image")
    print("First click in the left image, then the corresponding point in the right image")
    print("Press 'c' to clear all points and start over")
    print("Press 'r' to remove the last point")
    print("Press 'Enter' when done selecting points")
    
    # Initial display
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
                # Redraw remaining points
                display_img1 = img1.copy()
                for i, (px, py) in enumerate(points1):
                    cv.circle(display_img1, (px, py), 5, (0, 255, 0), -1)
                    cv.putText(display_img1, str(i+1), (px+5, py-5), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                param1['display_image'] = display_img1
                cv.imshow(window_name1, display_img1)
            
            if points2:
                points2.pop()
                # Redraw remaining points
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
    """Allow user to select blend boundary points on the image."""
    blend_points = []
    temp_image = image.copy()
    display_image = image.copy()
    
    def blend_mouse_callback(event, x, y, flags, param):
        nonlocal display_image
        
        if event == cv.EVENT_LBUTTONDOWN:
            param.append((x, y))
            # Update display image
            display_image = temp_image.copy()
            
            # Draw all points with connecting lines
            if len(param) > 1:
                points_array = np.array(param, dtype=np.int32)
                cv.polylines(display_image, [points_array], False, (0, 255, 0), 2)
            
            # Draw individual points
            for i, (px, py) in enumerate(param):
                cv.circle(display_image, (px, py), 5, (255, 0, 0), -1)
                cv.putText(display_image, str(i+1), (px+5, py-5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv.imshow(window_name, display_image)
            print(f"Blend boundary point {len(param)}: ({x}, {y})")
        
        elif event == cv.EVENT_MOUSEMOVE:
            # Show preview of line if we have at least one point
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
    """Create a blending mask from user-selected boundary points."""
    h, w = image_shape[:2]
    
    # Default mask if no points selected
    if boundary_points.shape[0] < 2:
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:, :w//2] = 1.0
        mask = cv.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        return cv.merge([mask, mask, mask])
    
    # Sort points by y-coordinate
    sorted_points = boundary_points[boundary_points[:,1].argsort()]
    
    # Add top and bottom points if needed
    if sorted_points[0,1] > 0:
        x_top = sorted_points[0,0]
        sorted_points = np.vstack(([[x_top, 0]], sorted_points))
    
    if sorted_points[-1,1] < h-1:
        x_bottom = sorted_points[-1,0]
        sorted_points = np.vstack((sorted_points, [[x_bottom, h-1]]))
    
    # Create closed polygon - left side
    poly_left = np.vstack(([[0,0]], [[0,h-1]], sorted_points[::-1]))
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.float32)
    cv.fillPoly(mask, [poly_left.astype(np.int32)], 1)
    
    # Apply Gaussian blur for smoother transition
    mask = cv.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    
    # Normalize mask values
    if mask.max() > 0:
        mask = mask / mask.max()
    
    # Create 3-channel mask
    mask = cv.merge([mask, mask, mask])
    
    return mask

def blend_images(img1, img2, mask):
    """Blend two images using Laplacian pyramid."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mask = mask.astype(np.float32)
    
    h, w = img1.shape[:2]
    img2 = cv.resize(img2, (w, h))
    mask = cv.resize(mask, (w, h))
    
    levels = 4
    pyr1 = laplacianPyramid(img1, levels)
    pyr2 = laplacianPyramid(img2, levels)
    mask_pyr = gaussianPyramid(mask, levels)
    
    blended_pyr = []
    for i in range(levels):
        mask_resized = cv.resize(mask_pyr[i], (pyr1[i].shape[1], pyr1[i].shape[0])).astype(np.float32)
        level1 = pyr1[i].astype(np.float32)
        level2 = pyr2[i].astype(np.float32)
        blended = level1 * mask_resized + level2 * (1.0 - mask_resized)
        blended_pyr.append(blended)
    
    return reconstruct(blended_pyr, levels)

# --- Bounding Box Shift Function ---
def bounding_box_shift(img):
    """
    Shift the mosaic image based on the bounding box of nonempty regions.
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
    Compute a 3x3 transformation (affine or perspective) matrix from the given source/destination points.
    Return both the matrix and the warped corners for bounding-box calculation.
    """
    # corners of the second image
    h, w = img.shape[:2]
    corners = np.array([
        [0,0], [w,0], [w,h], [0,h]
    ], dtype=np.float32).reshape(-1,1,2)

    if method == 'affine':
        # 2x3 matrix
        M_affine = cv.getAffineTransform(points_src[:3], points_dst[:3])
        # Convert to 3x3
        M_3x3 = np.eye(3, dtype=np.float32)
        M_3x3[:2] = M_affine
        # Warp corners
        warped_corners = cv.transform(corners, M_affine)
        return M_3x3, warped_corners
    else:
        M = cv.getPerspectiveTransform(points_src, points_dst)
        # Warp corners
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
    """Create a linear blending mask from left (1) to right (0)."""
    h, w = image_shape[:2]
    mask = np.tile(np.linspace(1, 0, w, dtype=np.float32), (h, 1))
    mask = cv.merge([mask, mask, mask])
    return mask

# def automatic_mosaic(img1, img2, patch_size=11, ncc_threshold=0.7):
    """
    Automatically compute point correspondences between img1 and img2 using Harris corners
    and normalized cross-correlation (NCC), then create a mosaic using better blending techniques.
    """
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    # Extract features using SIFT or ORB
    try:
        # Try SIFT first
        sift = cv.SIFT_create(nfeatures=500)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # Use FLANN for faster matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test for better matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    except:
        # Fall back to ORB if SIFT is not available
        orb = cv.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        # Use BFMatcher for ORB
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        try:
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        except:
            print("Matching failed. Trying basic matching.")
            matches = bf.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:50]  # Take top 50 matches
    
    if len(good_matches) < 4:
        print("Not enough good matches found.")
        return None, None
    
    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Compute homography with RANSAC for robust estimation
    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)
    
    if H is None:
        print("Could not compute homography.")
        return None, None
    
    # Calculate the dimensions of the output image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Get the corners of img2 - reshape correctly for perspectiveTransform
    corners2 = np.array([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Transform corners of img2
    warped_corners2 = cv.perspectiveTransform(corners2, H)
    
    # Find the min and max points to determine the size of the output image
    # Ensure corners1 is in the same format as warped_corners2
    corners1 = np.array([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Make sure all arrays have the same shape before vstack
    all_corners = np.vstack((corners1.reshape(-1, 2), warped_corners2.reshape(-1, 2)))
    
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    
    # Account for negative offsets
    offset_x = max(0, -min_x)
    offset_y = max(0, -min_y)
    
    # Calculate output dimensions
    output_width = max_x + offset_x
    output_height = max_y + offset_y
    
    # Create translation matrix
    translation_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Combine translation with homography
    warped_H = translation_matrix @ H
    
    # Create warped and translated images
    warped_img2 = cv.warpPerspective(img2, warped_H, (output_width, output_height))
    
    # Create a blank canvas for the first image
    img1_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
    
    # Place img1 on the canvas with the offset
    # Ensure we don't go out of bounds
    y_end = min(offset_y+h1, output_height)
    x_end = min(offset_x+w1, output_width)
    img1_canvas[offset_y:y_end, offset_x:x_end] = img1[:y_end-offset_y, :x_end-offset_x]
    
    # Create masks for each image (where pixels are non-zero)
    img1_mask = np.sum(img1_canvas, axis=2) > 0
    img2_mask = np.sum(warped_img2, axis=2) > 0
    
    # Find overlapping regions
    overlap = img1_mask & img2_mask
    
    # Create a gradual blend in overlapping areas
    result = np.zeros_like(img1_canvas)
    
    # Copy non-overlapping regions directly
    result[img1_mask & ~overlap] = img1_canvas[img1_mask & ~overlap]
    result[img2_mask & ~overlap] = warped_img2[img2_mask & ~overlap]
    
    # For overlapping regions, create a gradual blend
    if np.any(overlap):
        # Create distance maps from the boundaries of each image
        dist1 = cv.distanceTransform((~img2_mask).astype(np.uint8), cv.DIST_L2, 3)
        dist2 = cv.distanceTransform((~img1_mask).astype(np.uint8), cv.DIST_L2, 3)
        
        # Calculate weights based on distances
        weight = dist1 / (dist1 + dist2 + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Apply Gaussian blur for smoother transitions
        weight = cv.GaussianBlur(weight, (15, 15), 0)
        
        # Apply the weights to the overlapping regions
        for c in range(3):
            result[:,:,c][overlap] = (
                weight[overlap] * img1_canvas[:,:,c][overlap] + 
                (1 - weight[overlap]) * warped_img2[:,:,c][overlap]
            )
    
    # Clip to valid range and convert to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Remove unnecessary black borders
    gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_result, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the bounding rectangle of all contours combined
        all_contours = np.vstack([contour for contour in contours])
        x, y, w, h = cv.boundingRect(all_contours)
        
        # Crop the result to the bounding rectangle
        result = result[y:y+h, x:x+w]
    
    return result, (pts1, pts2)

def automatic_mosaic(img1, img2, patch_size=11, ncc_threshold=0.7):
    """
    Automatically compute point correspondences between img1 and img2 using Harris corners
    and normalized cross-correlation (NCC), then create a mosaic using better blending techniques.
    """
    import cv2 as cv
    import numpy as np

    # Convert images to grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    # Detect corners using Harris detector via goodFeaturesToTrack
    maxCorners = 1000
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    useHarrisDetector = True
    k = 0.04

    corners1 = cv.goodFeaturesToTrack(gray1, maxCorners, qualityLevel, minDistance,
                                      blockSize=blockSize, useHarrisDetector=useHarrisDetector, k=k)
    corners2 = cv.goodFeaturesToTrack(gray2, maxCorners, qualityLevel, minDistance,
                                      blockSize=blockSize, useHarrisDetector=useHarrisDetector, k=k)
    
    if corners1 is None or corners2 is None:
        print("No corners detected.")
        return None, None

    # Convert corners to integer (x, y) coordinates
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
        patch1 = gray1[y1-half_patch:y1+half_patch+1, x1-half_patch:x1+half_patch+1]
        best_ncc = -1.0
        best_pt2 = None
        
        for pt2 in corners2:
            x2, y2 = pt2
            # Ensure the patch from img2 is fully within the image boundaries
            if (x2 - half_patch < 0 or x2 + half_patch >= gray2.shape[1] or 
                y2 - half_patch < 0 or y2 + half_patch >= gray2.shape[0]):
                continue
            patch2 = gray2[y2-half_patch:y2+half_patch+1, x2-half_patch:x2+half_patch+1]
            
            # Compute normalized cross-correlation (NCC)
            patch1_mean = np.mean(patch1)
            patch2_mean = np.mean(patch2)
            patch1_std = np.std(patch1)
            patch2_std = np.std(patch2)
            if patch1_std < 1e-6 or patch2_std < 1e-6:
                continue
            ncc = np.sum((patch1 - patch1_mean) * (patch2 - patch2_mean)) / (patch_size * patch_size * patch1_std * patch2_std)
            
            if ncc > best_ncc:
                best_ncc = ncc
                best_pt2 = pt2
        
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
    
    # Determine the dimensions of the output mosaic
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Get corners of img2 and transform them using the computed homography
    corners2_rect = np.array([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners2 = cv.perspectiveTransform(corners2_rect, H)
    
    corners1_rect = np.array([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    all_corners = np.vstack((corners1_rect.reshape(-1, 2), warped_corners2.reshape(-1, 2)))
    
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    
    offset_x = max(0, -min_x)
    offset_y = max(0, -min_y)
    
    output_width = max_x + offset_x
    output_height = max_y + offset_y
    
    # Create the translation matrix and combine with homography
    translation_matrix = np.array([[1, 0, offset_x],
                                   [0, 1, offset_y],
                                   [0, 0, 1]], dtype=np.float32)
    warped_H = translation_matrix @ H
    
    # Warp img2 and prepare a canvas for img1
    warped_img2 = cv.warpPerspective(img2, warped_H, (output_width, output_height))
    img1_canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
    y_end = min(offset_y + h1, output_height)
    x_end = min(offset_x + w1, output_width)
    img1_canvas[offset_y:y_end, offset_x:x_end] = img1[:y_end-offset_y, :x_end-offset_x]
    
    # Create masks for each image
    img1_mask = np.sum(img1_canvas, axis=2) > 0
    img2_mask = np.sum(warped_img2, axis=2) > 0
    overlap = img1_mask & img2_mask
    
    # Blend the images together
    result = np.zeros_like(img1_canvas)
    result[img1_mask & ~overlap] = img1_canvas[img1_mask & ~overlap]
    result[img2_mask & ~overlap] = warped_img2[img2_mask & ~overlap]
    
    if np.any(overlap):
        dist1 = cv.distanceTransform((~img2_mask).astype(np.uint8), cv.DIST_L2, 3)
        dist2 = cv.distanceTransform((~img1_mask).astype(np.uint8), cv.DIST_L2, 3)
        weight = dist1 / (dist1 + dist2 + 1e-6)
        weight = cv.GaussianBlur(weight, (15, 15), 0)
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
    """Generate and save all required output images."""
    output_dir = "Template/PartB/output"
    os.makedirs(output_dir, exist_ok=True)
    
    image_pairs = [
        ('Template/PartB/image1_left.png', 'Template/PartB/image1_right.png', 'affine'),
        ('Template/PartB/image2_left.png', 'Template/PartB/image2_right.png', 'perspective'),
        ('Template/PartB/image3_left.png', 'Template/PartB/image3_right.png', 'affine'),
        ('Template/PartB/image4_left.png', 'Template/PartB/image4_right.png', 'perspective')
    ]
    
    img = cv.imread('Template/PartB/lena.png')
    if img is None:
        raise FileNotFoundError("Image not found: PartB/lena.png")
    
    # Example usage of the basic functions
    cv.imwrite(f'{output_dir}/convolved.png', convolve(img, np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])))
    cv.imwrite(f'{output_dir}/reduced.png', reduce(img))
    cv.imwrite(f'{output_dir}/expanded.png', expand(img))
    
    gaussian_pyr = gaussianPyramid(img, 3)
    for i, level in enumerate(gaussian_pyr):
        cv.imwrite(f'{output_dir}/gaussian_pyramid_level_{i}.png', level)
    
    laplacian_pyr = laplacianPyramid(img, 3)
    for i, level in enumerate(laplacian_pyr):
        cv.imwrite(f'{output_dir}/laplacian_pyramid_level_{i}.png', level)
    
    print("Testing reconstruction...")
    reconstructed = reconstruct(laplacianPyramid(img, 3), 3)
    cv.imwrite(f'{output_dir}/reconstructed.png', reconstructed)
    mse = np.mean((img.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    print(f"Reconstruction MSE: {mse:.4f}")
    
    error_file = os.path.join(output_dir, 'reconstruction_error.txt')
    with open(error_file, 'w') as f:
        f.write(f"Reconstruction Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Reconstruction Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}\n")
    print(f"Reconstruction error saved to: {error_file}")
    
    print("\nStarting image mosaicking (manual correspondences)...")
    
    for i, (left_path, right_path, preferred_method) in enumerate(image_pairs, 1):
        print(f"\nProcessing image pair {i}...")
        img1 = cv.imread(left_path)
        img2 = cv.imread(right_path)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not load images from {left_path} and {right_path}")
            continue
        
        print(f"\nCreating mosaic with {preferred_method} transformation...")
        mosaic_warped = mosaic_images([img1, img2], method=preferred_method)
        cv.imwrite(f'{output_dir}/mosaic_pair{i}_{preferred_method}.png', mosaic_warped)
        
        print("\nCreating mosaic without transformation...")
        mosaic_none = mosaic_images([img1, img2], method=None)
        cv.imwrite(f'{output_dir}/mosaic_pair{i}_none.png', mosaic_none)
    
    print("\nMosaic generation complete!")
    print("Generated 8 mosaicked images using manual correspondences.")

def generate_auto_mosaics():
    """Generate mosaicked images using automatic feature detection and matching."""
    output_dir = "Template/PartB/output_auto"
    os.makedirs(output_dir, exist_ok=True)
    
    image_pairs = [
        ('Template/PartB/image1_left.png', 'Template/PartB/image1_right.png'),
        ('Template/PartB/image2_left.png', 'Template/PartB/image2_right.png'),
        ('Template/PartB/image3_left.png', 'Template/PartB/image3_right.png'),
        ('Template/PartB/image4_left.png', 'Template/PartB/image4_right.png')
    ]
    
    for i, (img1_path, img2_path) in enumerate(image_pairs, 1):
        print(f"\nProcessing automatic mosaic for image pair {i}...")
        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not load images from {img1_path} and {img2_path}")
            continue
        
        mosaic, matches = automatic_mosaic(img1, img2)
        if mosaic is not None:
            cv.imwrite(f'{output_dir}/auto_mosaic_pair{i}.png', mosaic)
            print(f"Automatic mosaic for pair {i} saved.")
        else:
            print(f"Automatic mosaic for pair {i} could not be generated due to insufficient correspondences.")

if __name__ == "__main__":
    # Example usage:
    # generate_outputs()
    generate_auto_mosaics()
