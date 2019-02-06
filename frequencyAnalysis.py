# Grace Newman 4.1 Frequency Analysis

import cv2
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import skimage as ski


# Read image
img = cv2.imread('C:/Users/ciezcm15/Documents/Project1/window-06-06.JPG')
small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

# makes the image grayscale
grayImg = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:/Users/ciezcm15/Documents/Project1/grayImg.JPG', grayImg)


# Calculate the index of the middle column in the image
col = int(grayImg.shape[1]/2)
# Obtain the image data for this column
colData = small[0:grayImg.shape[0], col, 0]

# Plot the column data as a function
xvalues = np.linspace(0, len(colData) - 1, len(colData))
plt.plot(xvalues, colData, 'b')
plt.show()
# cv2.savefig('C:/Users/ciezcm15/Documents/Project1/colData.JPG', plt)


# Now plot cosine functions for frequencies -2pi/N * k
N = 128
xvalues = np.linspace(0, N, N)
for u in range(0, N):
     yvalues = np.sin(2 * np.pi * u / N * xvalues)
     plt.plot(xvalues, yvalues, 'r')
     # plt.savefig('/Users/ciezcm15/Documents/Project1/sin'+str(u)+'.png', bbox_inches='tight')
     # plt.clf()
     # print('saving image ' + str(u) + ' out of ' + str(N))

# Compute the 1-D Fourier transform of colData
F_colData = np.fft.fft(colData)

xvalues = np.linspace(-int(len(colData)/2), int(len(colData)/2)-1, len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.show()
# cv2.savefig('C:/Users/ciezcm15/Documents/Project1/FourierCoef.JPG', plt)

# Fourier Function complex function magnitudes
print(np.abs(F_colData))

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:grayImg.shape[0], 0:grayImg.shape[1]]
# create the figure
fig = plt.figure()
# ax = fig.gca(projection='3d')
#ax.plot_surface(xx, yy, grayImg, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
#plt.show()
# cv2.savefig('C:/Users/ciezcm15/Documents/Project1/2DFourier.JPG', plt)

# Take the 2D DFT and 3D plot the magnitude of the corresponding Fourier coefficients
F2_grayImg = np.fft.fft2(grayImg)
fig = plt.figure()
# ax = fig.gca(projection='3d')
Y = (np.linspace(-int(grayImg.shape[0]/2), int(grayImg.shape[0]/2)-1, grayImg.shape[0]))
X = (np.linspace(-int(grayImg.shape[1]/2), int(grayImg.shape[1]/2)-1, grayImg.shape[1]))
X, Y = np.meshgrid(X, Y)

# ax.plot_surface(X, Y, np.fft.fftshift(np.log(np.abs(F2_grayImg)+1)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
# plt.show()
# cv2.savefig('C:/Users/ciezcm15/Documents/Project1/logMagnitude.JPG', plt)

# Plot the magnitude and the log(magnitude + 1) as 2D images (view from the top)
magnitudeImage = np.fft.fftshift(np.abs(F2_grayImg))
magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
magnitudeImage = ski.img_as_ubyte(magnitudeImage)
# magnitudeImage = magnitudeImage / magnitudeImage.max()
# magnitudeImage = magnitudeImage.astype(np.uint8)

logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_grayImg)+1))
logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
# logMagnitudeImage = logMagnitudeImage.astype(np.uint8)

cv2.imshow('Magnitude plot', magnitudeImage)
cv2.imshow('Log Magnitude plot', logMagnitudeImage)
cv2.waitKey(0)

