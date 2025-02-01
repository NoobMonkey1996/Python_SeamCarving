# my own convolve function
# determines the value of central pixel by adding the weighted
#   values of all its neighbours together
# convolving can alter image via:
#       1. smooth
#       2. sharpen
#       3. intensify
#       4. enhance

# https://medium.com/@er_95882/convolution-image-filters-cnns-and-examples-in-python-pytorch-bd3f3ac5df9c

import sys
import imageio.v3 as iio 
import ipympl
import matplotlib.pyplot as plt
import skimage as ski
import cv2
import numpy as np
from numpy import asarray
import time

# start timing
start_time = time.time()

image0 = iio.imread("mario1.jpeg")
### acquire color values ad place kernel over it ###
# using 3x3 matrix kernel
def kernelSmooth():
    # the output pixel value using Convolution equation
    kernel = np.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]])
    return kernel

def kernelWeightedSmooth():
    kernel = np.array([[0,1,0],
                        [1,4,1],
                        [0,1,0]])
    return kernel

def kernelSharpen():
    kernel = np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
    return kernel

def kernelIntenseSharpen():
    kernel = np.array([[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])
    return kernel

imageArray = asarray(image0)
imageSize = imageArray.size

imHeight, imWidth, imChannels = imageArray.shape
# in the form of imageArray[height:width:0]; 0=R channel, 1=G channels, 2 = B channel
rChan = imageArray[:,:,0]
gChan = imageArray[:,:,1]
bChan = imageArray[:,:,2]

# fig, axRed = plt.subplots()
# axRed.imshow(rChan, cmap='Reds')
# fig.suptitle('Red Channel Only')
# plt.show(block=False)
# plt.pause(0.1)

# fig, axGreen = plt.subplots()
# axGreen.imshow(gChan, cmap='Greens')
# fig.suptitle('Green Channel Only')
# plt.show(block=False)
# plt.pause(0.1)

# fig, axBlue = plt.subplots()
# axBlue.imshow(bChan, cmap='Blues')
# fig.suptitle('Blue Channel Only')
# plt.show()
# plt.pause(0.1)

def processImage(image):
    image = cv2.imread(image)
    # turn image to grayscale
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image

def convolve2D(image, kernel, padding=0, strides=1):
    # cross correlation to preflip for later convolution
    kernel  = np.flipud(np.fliplr(kernel))
    
    # extract kernel and image shapes
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1] 
    xImgShape = image.shape[0] 
    yImgShape = image.shape[1]
    
    # compute matrix size of output image
    # apply size equation for each output dimension
    xOut = int(((xImgShape - xKernShape + 2 * padding) / strides + 1))
    yOut = int(((yImgShape - yKernShape + 2 * padding) / strides + 1))

    # create matrix with deduced dimensions
    output = np.zeros((xOut, yOut))
    
    # check if padding = 0, if so, avoid errors by ignoring following code
    if padding != 0:
        # multiply padding by 2 to apply even padding on all sides
        imagePadded = np.zeros((image.shape[0] + padding*2, 
                                image.shape[1] + padding*2))
        
        # replace inner portion of padded image with actual image
        imagePadded[int(padding):int(-1 * padding), 
                    int(padding):int(-1 * padding)] = image
        
    else: 
        imagePadded = image
    
    # convolution: iterate through image and apply element wise multiplication,
    #   and then sum it and set it equal to respective element in output array
    for y in range(image.shape[1]):
        if y > image.shape[1] - yKernShape:
            break       # exit once reaching bottom right of image matrix
        # only convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                # main convolution code, executing as per the equation
                try:
                    # only convolve if x has moved by specified Strides
                    output[x,y] = (kernel * imagePadded[x: x+xKernShape, y: y+yKernShape]).sum()
                except:
                    break
    return output
    
def main():
    # load in grayscale image
    image = processImage("mario1.jpeg")
    # kernel selected
    kernel = kernelIntenseSharpen()
    # convolve and save output
    output = convolve2D(image, kernel, padding=2)
    cv2.imwrite('2DConvolved.jpg', output)

# built-in variable that gets its value depending on how script is executed
# when script is run directly via (python script.py), __name__ is set to the modeul's name ('script')
if __name__ == '__main__':
    main()


# strategy for edge pixels
#   1. wrap image
#   2. ignore edge pixels and only compute for pixels with all neighbours
#   3. duplicate edge pixels 



print("%s seconds" % (time.time() - start_time))