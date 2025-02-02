import numpy as np
import cv2

def kernelgaussianBlur(kernel_size, sigma, mu):
    # Create a 1D Gaussian distribution
    x_Gauss = np.linspace(mu - 3 * sigma, mu + 3 * sigma, kernel_size)
    y_Gauss = np.exp(-0.5 * ((x_Gauss - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # Create a 2D Gaussian kernel by outer product
    kernel = np.outer(y_Gauss, y_Gauss)

    # Normalize the 2D kernel
    kernel /= kernel.sum()

    return kernel

def imageGray(image):
    image = cv2.imread(image)
    # Convert the image to grayscale
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image

def convolve2D(image, kernel, padding=0, strides=1):
    # Flip the kernel for convolution
    kernel = np.flipud(np.fliplr(kernel))

    # Extract kernel and image shapes
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Compute output dimensions
    xOut = int(((xImgShape - xKernShape + 2 * padding) / strides + 1))
    yOut = int(((yImgShape - yKernShape + 2 * padding) / strides + 1))

    # Initialize the output
    output = np.zeros((xOut, yOut))

    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - yKernShape:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - xKernShape:
                    break
                if x % strides == 0:
                    try:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break
    return output

def main():
    # Load the image
    image = imageGray("dragon.jpg")
    print("Image loaded and converted to grayscale.")

    # Select the kernel
    kernel = kernelgaussianBlur(25, 1, 0)
    print("Gaussian Blur Kernel:\n", kernel)

    # Perform convolution
    output = convolve2D(image, kernel, padding=2)
    print("Convolution completed.")

    return output

if __name__ == "__main__":
    output = main()
    cv2.imwrite('2DConvolvedDragon.jpg', output)
    print("Processing complete. Image saved as 2DConvolved.jpg.")

