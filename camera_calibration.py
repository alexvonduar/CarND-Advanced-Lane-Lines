import numpy as np
import os
import cv2
import pickle
import glob
import sys
import matplotlib.pyplot as plt

from helper_functions import save_image
from load_data import load_images

def do_camera_calibration(image_names, SAVE=""):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for image_name in image_names:
        img = cv2.imread(image_name)

        # save original images into save dir
        if SAVE != "":
            basename = os.path.basename(image_name)
            head, tail = os.path.split(SAVE)
            savename = os.path.join(head, basename)
            save_image(img, savename, append="original")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and save the chessboard corners
            if SAVE != "":
                basename = os.path.basename(image_name)
                head, tail = os.path.split(SAVE)
                savename = os.path.join(head, basename)
                chess_img = cv2.drawChessboardCorners(
                    img, (9, 6), corners, ret)
                # print("save :", savename)
                save_image(chess_img, savename, append="chess")

        else:
            print("can't fine chess board ", image_name)

    print("Found", len(imgpoints),
          "images with chessboard corners from", len(image_names), "images.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if SAVE != "":
        for image_name in image_names:
            img = cv2.imread(image_name)
            basename = os.path.basename(image_name)
            head, tail = os.path.split(SAVE)
            savename = os.path.join(head, basename)
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            # print("save :", savename)
            save_image(undist, savename, append="undistort")

    return mtx, dist


CAM_CAL_FILE = "calibration.pkl"
CAL_DIR = "camera_cal"

def save_matrix(path, mtx, dist):
    try:
        of = open(path, 'wb')
        save = {
            'MTX': mtx,
            'DIST': dist,
        }
        pickle.dump(save, of, pickle.HIGHEST_PROTOCOL)
        of.close()
    except Exception as e:
        print('Unable to save data to', path, ':', e)
        raise


def load_matrix(path):
    with open(path, mode='rb') as inf:
        calib = pickle.load(inf)

    return calib['MTX'], calib['DIST']

def cam_calib(cal_dir=CAL_DIR, cal_file=CAM_CAL_FILE, SAVE=""):
    if os.path.exists(cal_file):
        mtx, dist = load_matrix(cal_file)
    else:
        images = glob.glob(os.path.join(cal_dir, "calibration*.jpg"))
        mtx, dist = do_camera_calibration(images, SAVE)
        if save_matrix(cal_file, mtx, dist) == False:
            print("save pickle file failed")
    return mtx, dist

def undistort(image, cal_dir=CAL_DIR, cal_file=CAM_CAL_FILE, SAVE=""):
    if os.path.exists(cal_file):
        mtx, dist = load_matrix(cal_file)
    else:
        images = glob.glob(os.path.join(cal_dir, "calibration*.jpg"))
        mtx, dist = do_camera_calibration(images, SAVE)
        if save_matrix(cal_file, mtx, dist) == False:
            print("save pickle file failed")
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    return dst

def do_undistort(fname):
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dst = undistort(image)

    basename = os.path.basename(fname)
    head, ext = os.path.splitext(basename)
    savename = os.path.join('output_images', head + '_undistort' + ext)
    w = image.shape[1]
    h = image.shape[0]
    dpi = 96
    fig = plt.figure(figsize=(w/dpi, h/dpi))
    #plt.suptitle(fname)
    plt.subplot(211)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(212)
    plt.imshow(dst)
    plt.title('Undistort Image')
    plt.xlabel(fname)
    fig.tight_layout()
    fig.savefig(savename, dpi=dpi)
    plt.close()

def test_undistort(path):
    fnames = load_images(path)
    for fname in fnames:
        do_undistort(fname)


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        path = sys.argv.pop()
        test_undistort(path)