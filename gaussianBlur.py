import imageio.v3 as iio 
import ipympl
import matplotlib.pyplot as plt
import skimage as ski
import cv2

# https://datacarpentry.github.io/image-processing/06-blurring.html
# convolution, to convolve image with gaussian kernel

image  = iio.imread("mario1.jpeg")
fig, ax = plt.subplots()
ax.imshow(image)
fig.suptitle('Original picture')
plt.show(block=False)

# now comes the gaussian blur picture
sigma = 3.0
blurred = ski.filters.gaussian(
    image, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)

# sigma=(sigma, sigma) defines the sigma to use in ry- and cx- direction
# truncate pass the radius of the kernel in number of sigmas
# Gaussian function is defined from -infinity to +infinity
# hence need to set limit, truncate = 3.5 means kernel size = 2*sigma*3.5

# channel_axis specifies the dimension that contains the colour channels
# -1 refers to last position

fig, ay = plt.subplots()
ay.imshow(blurred)
fig.suptitle('Gaussian blurred')
plt.show(block=False)


### slice Intensity vs X axis graph
# find centre of image in Y axis

imHeight, imWidth = image.shape[:2]
print("Image height is ",imHeight, "Width is ", imWidth )

image_gray = ski.color.rgb2gray(image)
xmin,xmax = (0, image_gray.shape[1])
ymin = ymax = Y = image_gray.shape[0]//2    # // = integer division 
print(Y)
fig, bx = plt.subplots()
bx.imshow(image_gray, cmap='gray')
bx.plot([xmin, xmax], [ymin, ymax], color='red')
fig.suptitle('Intensity slice')
plt.show(block=False)


# pixel slice along Y axis
image_gray_sliceY = image_gray[Y,:]

# Normalise intensity to [0:255] unsigned integer
image_gray_sliceY = ski.img_as_ubyte(image_gray_sliceY)

fig, by = plt.subplots()
by.plot(image_gray_sliceY, color='red')
by.set_ylim(255,0)
fig.suptitle('Intensity vs X axis')
plt.show()
plt.pause(0.1)