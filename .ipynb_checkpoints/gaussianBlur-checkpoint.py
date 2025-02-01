import imageio.v3 as iio 
import ipympl
import matplotlib.pyplot as plt
import skimage as ski

# convolution, to convolve image with gaussian kernel

image  = iio.imread("mario1.jpeg")
fig, ax = plt.subplots()
ax.imshow(image)

plt.show()
