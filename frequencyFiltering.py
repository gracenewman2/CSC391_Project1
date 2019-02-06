# Grace Newman 5.1 Frequency Filtering

import cv2
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import skimage as ski


# Read image
img = cv2.imread('C:/Users/ciezcm15/Documents/Project1/window-00-03.JPG')
small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

# makes the image grayscale
grayImg = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', grayImg)

# Calculate the index of the middle column in the image
col = int(grayImg.shape[1]/2)
# Obtain the image data for this column
colData = small[0:grayImg.shape[0], col, 0]

# Plot the column data as a function
xvalues = np.linspace(0, len(colData) - 1, len(colData))
plt.plot(xvalues, colData, 'b')
# cv2.imwrite('C:/Users/ciezcm15/Documents/Project1/colData.JPG', plt)
plt.show()


# Compute the 1-D Fourier transform of colData
F_colData = np.fft.fft(colData)

xvalues = np.linspace(-int(len(colData)/2), int(len(colData)/2)-1, len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.show()

# Fourier Function complex function magnitudes
print(np.abs(F_colData))

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:grayImg.shape[0], 0:grayImg.shape[1]]
# create the figure
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, grayImg, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
# plt.show()


# Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients
F2_graySmall = np.fft.fft2(grayImg.astype(float))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
Y = (np.linspace(-int(grayImg.shape[0]/2), int(grayImg.shape[0]/2)-1, grayImg.shape[0]))
X = (np.linspace(-int(grayImg.shape[1]/2), int(grayImg.shape[1]/2)-1, grayImg.shape[1]))
X, Y = np.meshgrid(X, Y)


# U and V are arrays that give all integer coordinates in the 2-D plane
#  [-m/2 , m/2] x [-n/2 , n/2].
# Use U and V to create 3-D functions over (U,V)
U = (np.linspace(-int(grayImg.shape[0]/2), int(grayImg.shape[0]/2)-1, grayImg.shape[0]))
V = (np.linspace(-int(grayImg.shape[1]/2), int(grayImg.shape[1]/2)-1, grayImg.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(X*X + Y*Y)
# create x-points for plotting
xval = np.linspace(-int(grayImg.shape[1]/2), int(grayImg.shape[1]/2)-1, grayImg.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.25 * D.max()

# The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# and all D(u,v) where D(u,v) > 0 equal to 0
idealLowPass = D <= D0
idealHighPass = 1 - idealLowPass

# Filter our small grayscale image with the ideal lowpass filter
# 1. DFT of image
print(grayImg.dtype)
FTgraySmall = np.fft.fft2(grayImg.astype(float))
# 2. Butterworth filter is already defined in Fourier space
# 3. Elementwise product in Fourier space (notice fftshift of the filter)
FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealLowPass)
# FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealHighPass)
# 4. Inverse DFT to take filtered image back to the spatial domain
graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))

# idealHighPass = ski.img_as_ubyte(idealHighPass / idealHighPass.max())
idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())

cv2.imwrite("C:/Users/ciezcm15/Documents/Project1/idealLowPass.jpg", idealLowPass)
# cv2.imwrite("C:/Users/ciezcm15/Documents/Project1/idealHighPass.jpg", idealHighPass)
cv2.imwrite("C:/Users/ciezcm15/Documents/Project1/grayImageIdealLowpassFiltered.jpg", graySmallFiltered)

# Plot the ideal filter and then create and plot Butterworth filters of order
# n = 1, 2, 3, 4
plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
# plt.plot(xval, idealHighPass[int(idealHighPass.shape[0]/2), :], 'c--', label='ideal')

colors='brgkmc'
for n in range(1, 5):
    # Create Butterworth filter of order n
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    H1 = 1 - H
    # Apply the filter to the grayscaled image
    FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
    #vFTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H1)

    graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
    graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())

    cv2.imwrite("C:/Users/ciezcm15/Documents/Project1/grayImageButterworth-n" + str(n) + ".jpg", graySmallFiltered)
    # cv2.destroyAllWindows()

    H = ski.img_as_ubyte(H / H.max())
    # H1 = ski.img_as_ubyte(H1 / H1.max())

    cv2.imwrite("C:/Users/ciezcm15/Documents/Project1/butter-n" + str(n) + ".jpg", H)
    # cv2.imwrite("C:/Users/ciezcm15/Documents/Project1/butter-n" + str(n) + ".jpg", H1)

    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    # slice = H1[int(H1.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')

plt.show()
plt.savefig('C:/Users/ciezcm15/Documents/Project1/butterworthFilters1.jpg', bbox_inches='tight')

