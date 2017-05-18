import cv2
import numpy as np
import os
import sys
from load_data import load_images

def hls_process(name):
    img = cv2.imread(name)
    basename = os.path.basename(name)
    head, ext = os.path.splitext(basename)
    head = os.path.join('output_images', head)
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
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


