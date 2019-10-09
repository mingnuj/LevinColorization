import numpy as np
import cv2
from scipy import misc
import scipy
import matplotlib.pyplot as plt
from scipy.sparse import linalg
import colorsys
import math
import os
import sys


"""
Colorization using optimization

gray_path: Grayscale image
path: Image with masking
save_path: Path for saving
"""

# Input: grayscale image, masked image
gray_path = "./examples/example2.bmp"
path = "./examples/example2_marked.bmp"
save_path = "./examples/ex_result2.bmp"


# Read image with opencv-python
def read_image(image_path):
    img = cv2.imread(image_path).astype(np.float32) / 255.
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    _y, _u, _v = cv2.split(yuv_img)
    return img, _y, _u, _v


gray, y, g_u, g_v = read_image(gray_path)
colored, _, u, v = read_image(path)

# If it has color mask - True, Not - False
hints = abs(gray - colored).sum(2) > 0.01

h, w, _ = gray.shape
wrs = []

# All pixels
for j in range(h):
    for i in range(w):
        if not hints[j][i]:
            # Neighbors: pixels around selected pixel
            num_windows = 9
            neighbor_values = []
            # For all pixels in window (neighbor pixels)
            for win_j in range(max(0, j-1), min(j+2, h)):
                for win_i in range(max(0, i-1), min(i+2, w)):
                    # To make sparse matrix, saving index for [h*w, h*w] and intensity values.
                    row = j * w + i
                    col = win_j * w + win_i
                    # Selected pixel or not
                    if (i != win_i) | (j != win_j):
                        value = y[win_j, win_i]
                        neighbor_values.append((row, col, value))
                    else:
                        wrs.append((row, col, 1.))
            rows, cols, values = zip(*neighbor_values)
            values = list(values)
            values.append(y[j, i])

            # Calculate variance of neighbor pixels
            variance = np.mean((values - np.mean(values)) ** 2)
            sigma = variance * 0.6

            mgv = min((values - y[j][i])**2)
            if sigma < (-mgv / np.log(0.01)):
                sigma = -mgv / np.log(0.01)
            values[:] = values[:-1]
            if sigma < 0.000002:
                sigma = 0.000002

            # Calculate weight of neighbor pixels
            values = np.exp(-((values - y[j][i]) ** 2) / sigma)
            values = - (values / np.sum(values))

            for idx in range(len(values)):
                wrs.append((rows[idx], cols[idx], values[idx]))
        # If selected pixel is hint
        else:
            row = j * w + i
            col = j * w + i
            wrs.append((row, col, 1.))

row_idx, col_idx, wrs_value = zip(*wrs)

# Ax = b => Calculate colorization
A = scipy.sparse.csr_matrix((wrs_value, (row_idx, col_idx)), (w * h, w * h))
b = np.zeros((A.shape[0]))

color_copy_for_nonzero = hints.reshape(w*h).copy()
colored_idx = np.nonzero(color_copy_for_nonzero)

# U space solving
u_image = u.reshape(w * h)
print(u)
b[colored_idx] = u_image[colored_idx]
print(A.shape, b.shape)
new_vals_u = linalg.spsolve(A, b)
new_u = new_vals_u.reshape((h, w))

# V space solving
v_image = v.reshape(w * h)
b[colored_idx] = v_image[colored_idx]
new_vals_v = linalg.spsolve(A, b)
new_v = new_vals_v.reshape((h, w))

# Show result and saving
result = cv2.merge((y.astype(np.float32), new_u.astype(np.float32), new_v.astype(np.float32)))
result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)
cv2.imshow("result", result)
cv2.waitKey()
result = (result * 255).astype(np.int)
cv2.imwrite(save_path, result)
