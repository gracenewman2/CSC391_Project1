# Grace Newman 3.1 Spatial Filtering

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Read image
img = cv2.imread('C:/Users/ciezcm15/Documents/Project1/window-05-05.JPG') #the computer I got the Python to work on belongs to ciezcm15
small = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

# makes the image gray to do the edge detection
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BOX FILTER

windowLen = 3 #can be any size k

# w = np.ones((windowLen, windowLen), np.float32)
# w = w / w.sum()
# filt = np.zeros(small.shape, np.float64)  # array for filtered image

# Apply the filter to each channel
# filt[:, :, 0] = cv2.filter2D(small[:, :, 0], -1, w)
# filt[:, :, 1] = cv2.filter2D(small[:, :, 1], -1, w)
# filt[:, :, 2] = cv2.filter2D(small[:, :, 2], -1, w)

# print(filt.min())
# print(filt.max())

# filtered = filt / small.max()  # scale to [min/max, 1]
# filtered = filtered * 255 # scale to [min/max*255, 255]

Gaussian = cv2.GaussianBlur(img, (27, 27), 0) # k = 3, 9, 27
med = cv2.medianBlur(img, 27) # k = 3, 9, 27
edge = cv2.Canny(grayImg, 0, 200)

cv2.imshow('image', img)
cv2.imshow('filter', Gaussian)
cv2.imshow('median', med)
cv2.imshow('edges', edge)
# cv2.imshow('filtered', filtered.astype(np.uint8))
key = cv2.waitKey(0)

cv2.imwrite('C:/Users/ciezcm15/Documents/Project1/Ima.JPG', small)
cv2.imwrite('C:/Users/ciezcm15/Documents/Project1/Ima_Gaussian', Gaussian)
cv2.imwrite('C:/Users/ciezcm15/Documents/Project1/Ima_Median', med)
cv2.imwrite('C:/Users/ciezcm15/Documents/Project1/Ima_Edges', edge)
