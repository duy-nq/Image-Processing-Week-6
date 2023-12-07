import matplotlib.pyplot as plt
import numpy as np
from cv2 import medianBlur, dilate, erode

def readFromBinaryFile(filename):
    with open(filename, mode='rb') as file:
        fileContent = file.read()

    image = np.frombuffer(fileContent, dtype=np.uint8)
    image = np.reshape(image, (256, 256))

    return image

def window_filter(size):
    return np.ones((size, size))

def median_filter(f, image):
    pad = int((len(f)-1)/2)
    
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    for i in range(pad, len(padded_image)-pad):
        for j in range(pad, len(padded_image)-pad):
            window = np.median(padded_image[i-pad:i+pad+1, j-pad:j+pad+1])
            padded_image[i][j] = window
    
    return padded_image[1:-1, 1:-1]

def dilation(f, image):
    result = np.zeros(image.shape)
    
    pad = int((len(f) - 1) / 2)

    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    for i in range(pad, len(padded_image)-pad):
        for j in range(pad, len(padded_image)-pad):
            window = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            result[i-pad][j-pad] = window.max()

    return result

def erosion(f, image):
    result = np.zeros(image.shape)

    pad = int((len(f)-1)/2)

    padded_image = np.pad(image, pad, mode='constant', constant_values=255)

    for i in range(pad, len(padded_image)-pad):
        for j in range(pad, len(padded_image)-pad):
            window = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            result[i-pad][j-pad] = window.min()

    return result

def morphological_opening(filter, image):
    return dilation(filter, erosion(filter, image))

def morphological_closing(filter, image):
    return erosion(filter, dilation(filter, image))

def result(f, img1, img2):
    img1_mf = median_filter(f, img1)
    img2_mf = median_filter(f, img2)
    
    img1_mo = morphological_opening(f, img1)
    img2_mo = morphological_opening(f, img2)

    img1_mc = morphological_closing(f, img1)
    img2_mc = morphological_closing(f, img2)

    plt.subplot(241)
    plt.imshow(img1, cmap='gray')
    plt.title('Original')

    plt.subplot(242)
    plt.imshow(img1_mo, cmap='gray')
    plt.title('Morphological Opening')

    plt.subplot(243)
    plt.imshow(img1_mc, cmap='gray')
    plt.title('Morphological Closing')

    plt.subplot(244)
    plt.imshow(img1_mf, cmap='gray')
    plt.title('Median Filter')

    plt.subplot(245)
    plt.imshow(img2, cmap='gray')
    plt.title('Original')

    plt.subplot(246)
    plt.imshow(img2_mo, cmap='gray')
    plt.title('Morphological Opening')

    plt.subplot(247)
    plt.imshow(img2_mc, cmap='gray')
    plt.title('Morphological Closing')

    plt.subplot(248)
    plt.imshow(img2_mf, cmap='gray')
    plt.title('Median Filter')

    plt.show()

def main():
    cam9 = readFromBinaryFile('camera9bin.sec')
    cam99 = readFromBinaryFile('camera99bin.sec')

    sqr_3x3 = window_filter(3)

    result(sqr_3x3, cam9, cam99)

if __name__ == '__main__':
    main()
    

