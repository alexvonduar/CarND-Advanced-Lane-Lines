import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from load_data import load_images

def hls_process(name):
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    basename = os.path.basename(name)
    head, ext = os.path.splitext(basename)
    head = os.path.join('output_images', head)
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    h_channel = hls[:, :, 0]
    cv2.imwrite(head + '_s' + ext, s_channel)
    cv2.imwrite(head + '_l' + ext, l_channel)
    cv2.imwrite(head + '_h' + ext, h_channel)

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 255
    cv2.imwrite(head + '_sthresh' + ext, s_binary)

    l_thresh_min = 25
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min)] = 255
    cv2.imwrite(head + '_lthresh' + ext, l_binary)

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(l_binary)
    combined_binary[(s_binary > 0) & (l_binary > 0)] = 255
    cv2.imwrite(head + '_comb' + ext, combined_binary)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(sxbinary)))
    color_binary2 = np.dstack((sxbinary, combined_binary, np.zeros_like(sxbinary)))

    basename = os.path.basename(name)
    head, ext = os.path.splitext(basename)
    savename = os.path.join('output_images', head + '_binary' + ext)
    w = img.shape[1]
    h = img.shape[0]
    dpi = 96
    fig = plt.figure(figsize=(3 * w/dpi, 4 * h / dpi))
    plt.suptitle(name)
    plt.subplot(432)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(434)
    plt.imshow(s_channel, cmap='gray')
    plt.title('S Channel')
    plt.subplot(435)
    plt.imshow(l_channel, cmap='gray')
    plt.title('L Channel')
    plt.subplot(433)
    plt.imshow(scaled_sobel, cmap='gray')
    plt.title('Sobel')
    plt.subplot(436)
    plt.imshow(sxbinary, cmap='gray')
    plt.title('Thresholded Sobel')
    plt.subplot(437)
    plt.imshow(s_binary, cmap='gray')
    plt.title('Thresholded S')
    plt.subplot(438)
    plt.imshow(l_binary, cmap='gray')
    plt.title('Thresholded L')
    plt.subplot(439)
    plt.imshow(combined_binary, cmap='gray')
    plt.title('Thresholde S & L')
    plt.subplot(4,3,10)
    plt.imshow(color_binary)
    plt.title('S + Sobel')
    plt.subplot(4,3,11)
    plt.imshow(color_binary2)
    plt.title('(S & L) + Sobel')
    fig.tight_layout()
    fig.savefig(savename, dpi=dpi)
    plt.close()

    return combined_binary

def test_hls_process(path):
    fnames = load_images(path)

    for fname in fnames:
        hls_process(fname)

if __name__ == "__main__":
    path = "test_images"
    if (len(sys.argv) > 1):
        path = sys.argv.pop()

    test_hls_process(path)


