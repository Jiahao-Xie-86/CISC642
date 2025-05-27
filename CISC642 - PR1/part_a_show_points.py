import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read correspondances
pts_original_affine = np.loadtxt('PartA\\correspondances\\affine\\water.csv', delimiter=',', skiprows=1).astype(np.float32)

pts_original = pts_original_affine[:, 0:2]
pts_affine = pts_original_affine[:, 2:4]

# Read images
img_original = cv.imread('PartA\\original\\water.jpg')
img_affine = cv.imread('PartA\\affine\\water.jpg')

# BGR to RGB
img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
img_affine = cv.cvtColor(img_affine, cv.COLOR_BGR2RGB)

# Display original image
plt.scatter(x=pts_original[:, 0], y=pts_original[:, 1], c='r', s=40)
plt.imshow(img_original)
plt.show()
plt.clf()

# Display affine image
plt.scatter(x=pts_affine[:, 0], y=pts_affine[:, 1], c='r', s=40)
plt.imshow(img_affine)
plt.show()
plt.clf()
