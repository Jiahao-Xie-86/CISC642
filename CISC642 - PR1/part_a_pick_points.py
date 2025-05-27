import cv2 as cv
import numpy as np
import os
import pandas as pd

# Setup directories
current_dir = os.path.dirname(os.path.abspath(__file__))
original_dir = os.path.join(current_dir, "PartA", "original")
affine_dir = os.path.join(current_dir, "PartA", "affine")
perspective_dir = os.path.join(current_dir, "PartA", "perspective")
correspondences_dir = os.path.join(current_dir, "PartA", "correspondances")

# Create correspondence directories if they don't exist
os.makedirs(os.path.join(correspondences_dir, "affine"), exist_ok=True)
os.makedirs(os.path.join(correspondences_dir, "perspective"), exist_ok=True)

# Initialize lists to store points
original_points = []
affine_points = []
perspective_points = []

def original(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        original_points.append((x, y))
        print(f'[Original] ({x}, {y})')
        # Draw circle at clicked point
        cv.circle(original_image, (x, y), 3, (0, 255, 0), -1)
        cv.imshow('Original Image', original_image)

def affine(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        affine_points.append((x, y))
        print(f'[Affine] ({x}, {y})')
        # Draw circle at clicked point
        cv.circle(affine_image, (x, y), 3, (0, 255, 0), -1)
        cv.imshow('Affine Image', affine_image)

def perspective(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        perspective_points.append((x, y))
        print(f'[Perspective] ({x}, {y})')
        # Draw circle at clicked point
        cv.circle(perspective_image, (x, y), 3, (0, 255, 0), -1)
        cv.imshow('Perspective Image', perspective_image)

def save_correspondences():
    """Save the collected points to CSV files."""
    # Save affine correspondences
    if len(original_points) == len(affine_points):
        affine_data = np.hstack((np.array(original_points), np.array(affine_points)))
        affine_df = pd.DataFrame(affine_data, columns=['x_og', 'y_og', 'x_transformed', 'y_transformed'])
        affine_csv = os.path.join(correspondences_dir, "affine", "lena.csv")
        affine_df.to_csv(affine_csv, index=False)
        print(f"Saved affine correspondences to {affine_csv}")
    else:
        print("Number of points in original and affine images don't match!")

    # Save perspective correspondences
    if len(original_points) == len(perspective_points):
        perspective_data = np.hstack((np.array(original_points), np.array(perspective_points)))
        perspective_df = pd.DataFrame(perspective_data, columns=['x_og', 'y_og', 'x_transformed', 'y_transformed'])
        perspective_csv = os.path.join(correspondences_dir, "perspective", "lena.csv")
        perspective_df.to_csv(perspective_csv, index=False)
        print(f"Saved perspective correspondences to {perspective_csv}")
    else:
        print("Number of points in original and perspective images don't match!")

# Load images
original_image = cv.imread(os.path.join(original_dir, 'computer_add_10.png'))
affine_image = cv.imread(os.path.join(affine_dir, 'computer_add_10.png'))
perspective_image = cv.imread(os.path.join(perspective_dir, 'computer_add_10.png'))

if original_image is None or affine_image is None or perspective_image is None:
    print("Error: Could not load one or more images!")
    exit()

# Create windows and set callbacks
cv.namedWindow('Original Image')
cv.namedWindow('Affine Image')
cv.namedWindow('Perspective Image')
cv.setMouseCallback('Original Image', original)
cv.setMouseCallback('Affine Image', affine)
cv.setMouseCallback('Perspective Image', perspective)

# Show images
cv.imshow('Original Image', original_image)
cv.imshow('Affine Image', affine_image)
cv.imshow('Perspective Image', perspective_image)

print("\nInstructions:")
print("AFFINE MODE (First):")
print("1. Click EXACTLY 3 points in the Original image (green)")
print("2. Click EXACTLY 3 corresponding points in the Affine image (green)")
print("3. Press 'm' to switch to perspective mode")
print("\nPERSPECTIVE MODE (Second):")
print("4. Click EXACTLY 4 points in the Original image (red)")
print("5. Click EXACTLY 4 corresponding points in the Perspective image (red)")
print("\nCONTROLS:")
print("- Press 'm' to switch between affine and perspective modes")
print("- Press 'r' to reset current mode's points")
print("- Press 's' to save the correspondences")
print("- Press 'q' to quit")

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        save_correspondences()
    elif key == ord('q'):
        break

cv.destroyAllWindows()
print("\nPoints collected:")
print(f"Original: {len(original_points)}")
print(f"Affine: {len(affine_points)}")
print(f"Perspective: {len(perspective_points)}")
