import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_images
from helper_functions import save_image
from camera_calibration import undistort

#src = np.float32([[500, 482], [800, 482], [1240, 720], [50, 720]])
#dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])
src = np.float32([[545, 480], [742, 480], [1030, 660], [280, 660]])
# dst = np.float32([[350, 0], [930, 0], [930, 720], [350, 720]])
dst = np.float32([[220, 100], [1060, 100], [1060, 700], [220, 700]])

def forward_matrix():
    return cv2.getPerspectiveTransform(src, dst)

def backward_matrix():
    return cv2.getPerspectiveTransform(dst, src)


def persp_trans_forward(img):
    size = (img.shape[1], img.shape[0])
    M = forward_matrix()
    dst = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    return dst

def persp_trans_backward(img):
    size = (img.shape[1], img.shape[0])
    M = backward_matrix()
    dst = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    return dst

def draw_src_rectangle(img):
    cv2.line(img, (src[0][0], src[0][1]),
             (src[1][0], src[1][1]), (255, 0, 0), thickness=2)
    cv2.line(img, (src[1][0], src[1][1]),
             (src[2][0], src[2][1]), (255, 0, 0), thickness=2)
    cv2.line(img, (src[2][0], src[2][1]),
             (src[3][0], src[3][1]), (255, 0, 0), thickness=2)
    cv2.line(img, (src[3][0], src[3][1]),
             (src[0][0], src[0][1]), (255, 0, 0), thickness=2)
    return img


def draw_dst_rectangle(img):
    cv2.line(img, (dst[0][0], dst[0][1]),
             (dst[1][0], dst[1][1]), (255, 0, 0), thickness=2)
    cv2.line(img, (dst[1][0], dst[1][1]),
             (dst[2][0], dst[2][1]), (255, 0, 0), thickness=2)
    cv2.line(img, (dst[2][0], dst[2][1]),
             (dst[3][0], dst[3][1]), (255, 0, 0), thickness=2)
    cv2.line(img, (dst[3][0], dst[3][1]),
             (dst[0][0], dst[0][1]), (255, 0, 0), thickness=2)
    return img


def validate_perspective_transform(inputdir, outputdir):
    fnames = load_images(inputdir)

    for fname in fnames:
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = undistort(image)

        basename = os.path.basename(fname)
        savename = os.path.join(outputdir, basename)

        src = np.copy(image)
        src = draw_src_rectangle(src)
        save_image(src, savename, 'persp_src')

        dst = persp_trans_forward(image)
        dst = draw_dst_rectangle(dst)
        save_image(dst, savename, 'persp_dst')

        basename = os.path.basename(fname)
        head, ext = os.path.splitext(basename)
        savename = os.path.join('output_images', head + '_perspective' + ext)
        w = image.shape[1]
        h = image.shape[0]
        dpi = 96
        fig = plt.figure(figsize=(w/dpi, h/dpi))
        plt.suptitle(fname)
        plt.subplot(121)
        plt.imshow(src)
        plt.title('Undistort Image')
        plt.subplot(122)
        plt.imshow(dst)
        plt.title('Perspective Transform')
        #plt.xlabel(fname)
        fig.tight_layout()
        fig.savefig(savename, dpi=dpi)
        plt.close()

if __name__ == "__main__":
    test_dir = "test_images"
    output_dir = "output_images"
    if len(sys.argv) == 1:
        print("use default test dir:", test_dir,
              " default output dir:", output_dir)
    else:
        test_dir = sys.argv.pop()
        output_dir = sys.argv.pop()

    validate_perspective_transform(test_dir, output_dir)
