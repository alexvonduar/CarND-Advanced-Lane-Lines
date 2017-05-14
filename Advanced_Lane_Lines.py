import glob
import os
import pickle
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

import cv2

#%matplotlib qt

# helper functions and global defs
TMP_DIR = "output_images"
CAL_DIR = "camera_cal"
# OUT_DIR = "output_images"
src = np.float32([[490, 482], [810, 482], [1250, 720], [40, 720]])
dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])


def save_result(img, path, append=None):
    if path != "":
        head, tail = os.path.split(path)
        name, ext = os.path.splitext(tail)
        if append != None:
            # print("save name :", path, append)
            append = "_" + append
        savename = os.path.join(head, name + append + ext)
        cv2.imwrite(savename, img)
        # print("save file :", savename)


def save_plot(data, path, append=None):
    if path != "":
        head, tail = os.path.split(path)
        name, ext = os.path.splitext(tail)
        if append != None:
            # print("save name :", path, append)
            append = "_" + append
        savename = os.path.join(head, name + append + ext)
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(data)
        fig.savefig(savename)   # save the figure to file
        plt.close(fig)    # close the figure

# do camera calibration from input images


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
            save_result(img, savename, append="original")

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
                save_result(chess_img, savename, append="chess")

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
            save_result(undist, savename, append="undistort")

    return mtx, dist

# dist, mtx = camera_calibration()


CAM_CAL_FILE = "calibration.pkl"


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


def gen_binary_images(img, mtx, dist, SAVE=""):
    # save original image
    if SAVE == "" and left_lane.savename != "":
        savename = os.path.join(left_lane.savepath, left_lane.savename)
    else:
        savename = SAVE
    save_result(img, savename, "original")

    # undistort image
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    save_result(dst, savename, "undistort")

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(dst, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what
    # might work better
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 190
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can
    # see as different colors
    color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(sxbinary)))
    color_binary[(sxbinary == 1), 0] = 255
    color_binary[(s_binary == 1), 1] = 255
    cv2.line(color_binary, (src[0][0], src[0][1]), (src[1][0], src[1][1]), (0, 0, 255))
    cv2.line(color_binary, (src[1][0], src[1][1]), (src[2][0], src[2][1]), (0, 0, 255))
    cv2.line(color_binary, (src[2][0], src[2][1]), (src[3][0], src[3][1]), (0, 0, 255))
    cv2.line(color_binary, (src[3][0], src[3][1]), (src[0][0], src[0][1]), (0, 0, 255))
    save_result(color_binary, savename, "color_stack")

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255

    save_result(combined_binary, savename, "combined")
    return combined_binary

def perspective_transform(img, src, dst):
    size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    dst = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    return dst

# Define a class to receive the characteristics of each line detection


class Line():
    def __init__(self, default_path=""):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # debug
        self.debug = False
        # number of frames processed
        self.num_frames = 0
        # frame window
        self.frames_per_window = 3
        # debug name
        self.savename = "{0:08d}".format(self.num_frames) + ".jpg"
        # debug save path
        self.savepath = default_path

    def enableDebug(self, path=""):
        self.debug = True
        if path != "":
            self.savepath = path

    def update(self, x, y):
        self.savename = "{0:08d}".format(self.num_frames) + ".jpg"

        self.num_frames += 1


def detect_lanes(image, prev_lanes=None, save_path=""):
    if save_path == "" and left_lane.savepath != "":
        savename = os.path.join(left_lane.savepath, left_lane.savename)
    else:
        savename = save_path

    if True:  # not prev_lanes:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[int(image.shape[0] / 2):, :], axis=0)
        top_histogram = np.sum(image[:int(image.shape[0] / 2), :], axis=0)
        save_plot(histogram, savename, "bottom_hist")
        save_plot(top_histogram, savename, "top_hist")
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        left_search_center = []
        right_search_center = []
        left_search_center.append(leftx_current)
        right_search_center.append(rightx_current)

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                left_search_center.append(leftx_current)
            else:
                left_search_center.append(None)
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                right_search_center.append(rightx_current)
            else:
                right_search_center.append(None)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    else:
        left_fit = prev_lanes[0]
        right_fit = prev_lanes[1]
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        left_lane_center = left_fit[0] * \
            (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2]
        left_lane_boarder0 = left_lane_center - margin
        left_lane_boarder1 = left_lane_center + margin
        left_lane_inds = ((nonzerox > left_lane_boarder0) &
                          (nonzerox < left_lane_boarder1))

        right_lane_center = right_fit[0] * \
            (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2]
        right_lane_boarder0 = right_lane_center - margin
        right_lane_boarder1 = right_lane_center + margin
        right_lane_inds = ((nonzerox > right_lane_boarder0)
                           & (nonzerox < right_lane_boarder1))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Measure Radius of Curvature for each lane line
    ym_per_pix = 30. / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * np.max(lefty) + left_fit_cr[1])**2)**1.5) \
        / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(lefty) + right_fit_cr[1])**2)**1.5) \
        / np.absolute(2 * right_fit_cr[0])

    # Calculate the position of the vehicle
    rightx_int = right_fit[0] * 720**2 + right_fit[1] * 720 + right_fit[2]
    leftx_int = left_fit[0] * 720**2 + left_fit[1] * 720 + left_fit[2]
    center = abs((1280 / 2) - ((rightx_int + leftx_int) / 2))

    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((image, image, image)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Draw the fit lines
    left_x = left_fitx
    left_x[left_x < 0] = 0
    left_x[left_x >= 1280] = 1279
    right_x = right_fitx
    right_x[right_x < 0] = 0
    right_x[right_x >= 1280] = 1279
    l_points = np.squeeze(
        np.array(np.dstack((left_x, ploty)), dtype='int32'))
    r_points = np.squeeze(
        np.array(np.dstack((right_x, ploty)), dtype='int32'))
    out_img[l_points[:, 1], l_points[:, 0]] = [0, 255, 255]
    out_img[r_points[:, 1], r_points[:, 0]] = [0, 255, 255]

    # Draw the search box
    for window in range(nwindows):
        if left_search_center[window] != None:
            win_x0 = left_search_center[window] - margin
            win_x1 = left_search_center[window] + margin
            win_y0 = image.shape[0] - (window + 1) * window_height
            win_y1 = image.shape[0] - window * window_height
            cv2.rectangle(out_img, (win_x0, win_y0), (win_x1, win_y1), (0, 255, 255))
        if right_search_center[window] != None:
            win_x0 = right_search_center[window] - margin
            win_x1 = right_search_center[window] + margin
            win_y0 = image.shape[0] - (window + 1) * window_height
            win_y1 = image.shape[0] - window * window_height
            cv2.rectangle(out_img, (win_x0, win_y0), (win_x1, win_y1), (0, 255, 255))

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    left_lane.update(left_fitx, ploty)
    right_lane.update(right_fitx, ploty)

    # print("save lane image: ", savename)
    save_result(out_img, savename, "lane")

    return left_fit, right_fit


# Define conversions in x and y from pixels space to meters
IMG_WIDTH = 1280
IMG_HEIGHT = 720
LANE_WIDTH_PX = 640
YM_PER_PX = 30 / IMG_HEIGHT  # meters per pixel in y dimension
XM_PER_PX = 3.7 / LANE_WIDTH_PX  # meters per pixel in x dimension

# Calculate distance in meters from center of lane

def dist_from_center(left_fitx, right_fitx):
    # Calculate distance from center
    # x position of left line at y = 720
    left_x = left_fitx[-1]
    right_x = right_fitx[-1]
    center_x = left_x + ((right_x - left_x) / 2)
    return ((IMG_WIDTH / 2) - center_x) * XM_PER_PX

# Calculate the average curvature radius from the detected fitting
# parameters of left & right curves

def get_curverad(ploty, left_fitx, right_fitx):
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * YM_PER_PX, left_fitx * XM_PER_PX, 2)
    right_fit_cr = np.polyfit(ploty * YM_PER_PX, right_fitx * XM_PER_PX, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * YM_PER_PX + left_fit_cr[1]) ** 2) ** 1.5) \
        / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * YM_PER_PX + right_fit_cr[1]) ** 2) ** 1.5) \
        / np.absolute(2 * right_fit_cr[0])
    return (left_curverad + right_curverad) / 2


def generate_plot(img, lfit, rfit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = lfit[0] * ploty ** 2 + lfit[1] * ploty + lfit[2]
    right_fitx = rfit[0] * ploty ** 2 + rfit[1] * ploty + rfit[2]
    return left_fitx, ploty, right_fitx


def plot_lane(input_image, left_fitx, ploty, right_fitx):
    l_points = np.squeeze(
        np.array(np.dstack((left_fitx, ploty)), dtype='int32'))
    r_points = np.squeeze(
        np.array(np.dstack((right_fitx, ploty)), dtype='int32'))
    out_img = np.zeros_like(input_image)
    points_rect = np.concatenate((r_points, l_points[::-1]), 0)
    cv2.fillPoly(out_img, [points_rect], (0, 255, 0))
    cv2.polylines(out_img, [l_points], False, (255, 0, 0), 15)
    cv2.polylines(out_img, [r_points], False, (0, 0, 255), 15)
    return out_img


prev_lanes = None


def process_image(img, lanes=None, SAVE=""):
    if SAVE == "" and left_lane.savepath != "":
        savename = os.path.join(left_lane.savepath, left_lane.savename)
    else:
        savename = SAVE

    mtx, dist = cam_calib(SAVE=savename)

    # print("process :", fname)
    bin_img = gen_binary_images(img, mtx, dist, savename)

    warp_img = perspective_transform(bin_img, src, dst)
    save_result(warp_img, savename, "warp")

    lanes = detect_lanes(warp_img, prev_lanes=None, save_path=savename)

    left_fitx, ploty, right_fitx = generate_plot(
        warp_img, lanes[0], lanes[1])

    out_img = plot_lane(img, left_fitx, ploty, right_fitx)

    # Distance from center
    dist_x = dist_from_center(left_fitx, right_fitx)
    # Radius of curvature
    curverad = get_curverad(ploty, left_fitx, right_fitx)
    # Draw lane into original image, first do inverse perspective tranformation
    out_img = perspective_transform(out_img, dst, src)
    out_img = cv2.addWeighted(img, .5, out_img, .5, 0.0, dtype=0)
    cv2.putText(out_img, "Radius: %.2fm" % curverad, (20, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
    cv2.putText(out_img, "Distance from center: %.2fm" %
                (dist_x), (20, 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))

    # print("save name :", save_name)
    save_result(out_img, savename, "final")

    return out_img

def process_video_image(img, lanes=None, SAVE=""):
    r,g,b = cv2.split(img)
    img = cv2.merge([b,g,r])
    img = process_image(img, lanes=lanes, SAVE=SAVE)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    return img

def process_test_images():
    images = glob.glob('test_images/*.jpg')
    save_path = "output_images"

    for fname in images:
        img = cv2.imread(fname)
        head, tail = os.path.split(fname)
        save_name = os.path.join(save_path, tail)
        prev_lanes = None
        process_image(img, lanes=None, SAVE=save_name)


left_lane = Line()
right_lane = Line()

process_test_images()


src = np.float32([[595, 451], [680, 451], [233, 720], [1067, 720]])
dst = np.float32([[350, 0], [930, 0], [350, 720], [930, 720]])


def load_test_video(file_name='test_video.mp4'):
    vimages = []
    vframes = []
    count = 0
    clip = VideoFileClip(file_name)
    for img in clip.iter_frames(progress_bar=True):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vimages.append(img)
        vframes.append("%s - %d" % (file_name, count))
        count += 1

    return vimages, vframes

left_lane = Line(os.path.join(TMP_DIR, "video"))
right_lane = Line(os.path.join(TMP_DIR, "video"))

input_video = "project_video.mp4"
output_video = "project_lane.mp4"

'''
clip1 = VideoFileClip(input_video)
output_clip = clip1.fl_image(process_video_image)
output_clip.write_videofile(output_video, audio=False)
'''
