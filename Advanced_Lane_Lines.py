import glob
import os
import pickle
import sys
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from scipy import optimize

import cv2
from gaussian_fit import gaussian, gaussian_sfit
from helper_functions import save_image
from helper_functions import save_hist
from camera_calibration import cam_calib
from perspective import persp_trans_backward
from perspective import persp_trans_forward

#%matplotlib qt

# helper functions and global defs
TMP_DIR = "output_images"

# OUT_DIR = "output_images"


def gen_binary_images(img, mtx, dist, SAVE=""):
    # save original image
    if SAVE == "" and left_lane.savepath != "":
        savename = os.path.join(left_lane.savepath, left_lane.savename)
    else:
        savename = SAVE
    save_image(img, savename, "original")

    # undistort image
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    save_image(dst, savename, "undistort")

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(dst, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for
    # the lane lines
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

    # Threshold luma channel
    l_thresh_min = 30
    #l_binary = np.zeros_like(l_channel)
    #l_binary[l_channel >= l_thresh_min] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max) & (l_channel >= l_thresh_min)] = 1


    # Stack each channel to view their individual contributions in green and
    # blue respectively
    # This returns a stack of the two binary images, whose components you can
    # see as different colors
    color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(sxbinary)))
    color_binary[(sxbinary == 1), 0] = 255
    color_binary[(s_binary == 1), 1] = 255
    save_image(color_binary, savename, "color_stack")

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255

    save_image(combined_binary, savename, "combined")
    return combined_binary


# Define a class to receive the characteristics of each line detection


class Lane():
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
        # number of history result saved
        self.num_history = 0
        # maximum number of history save
        self.MAX_NUM_HISTORY = 7
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
        self.detected = True

        self.allx = x
        self.ally = y

        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        self.recent_xfitted.append(self.current_fit)
        self.num_history += 1

        if (self.num_history > self.MAX_NUM_HISTORY):
            self.recent_xfitted.pop()

        self.best_fit = np.mean(self.recent_xfitted, axis=0)

        self.num_frames += 1


def detect_lanes(image, prev_lanes=None, save_path=""):
    global left_lane
    global right_lane
    if save_path == "" and left_lane.savepath != "":
        savename = os.path.join(left_lane.savepath, left_lane.savename)
    else:
        savename = save_path

    # Choose the number of sliding windows
    nwindows = 9
    left_center_line = []
    right_center_line = []
    if left_lane.detected == False:
        # Take a histogram of the bottom half of the image
        bot_histogram = np.sum(image[int(image.shape[0] / 2):, :], axis=0)
        top_histogram = np.sum(image[:int(image.shape[0] / 2), :], axis=0)
        save_hist(bot_histogram, savename, "bottom_hist")
        save_hist(top_histogram, savename, "top_hist")
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((image, image, image)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base, rightx_base = gen_from_hist(
            bot_histogram, top_histogram)

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
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        half_nwindow = int(np.floor(nwindows / 2))

        center_y_low = half_nwindow * window_height
        center_y_hi = center_y_low
        if (nwindows % 2 != 0):
            left_center_line.append(leftx_current)
            right_center_line.append(rightx_current)
            center_y_hi = (half_nwindow + 1) * window_height
            win_y_low = center_y_low
            win_y_high = center_y_hi
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
                left_center_line.append(leftx_current)
            else:
                left_center_line.append(None)
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                right_center_line.append(rightx_current)
            else:
                right_center_line.append(None)

        # Step through the windows one by one
        top_leftx_current = leftx_current
        bot_leftx_current = leftx_current
        top_rightx_current = rightx_current
        bot_rightx_current = rightx_current
        for window in range(half_nwindow):
            # Top
            # Identify window boundaries in x and y (and right and left)
            win_y_low = center_y_low - (window + 1) * window_height
            win_y_high = center_y_low - window * window_height
            win_xleft_low = top_leftx_current - margin
            win_xleft_high = top_leftx_current + margin
            win_xright_low = top_rightx_current - margin
            win_xright_high = top_rightx_current + margin
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
                top_leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                left_center_line.append(top_leftx_current)
            else:
                left_center_line.append(None)
            if len(good_right_inds) > minpix:
                top_rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                right_center_line.append(top_rightx_current)
            else:
                right_center_line.append(None)

            # bottom
            # Identify window boundaries in x and y (and right and left)
            win_y_low = center_y_hi + window * window_height
            win_y_high = center_y_hi + (window + 1) * window_height
            win_xleft_low = bot_leftx_current - margin
            win_xleft_high = bot_leftx_current + margin
            win_xright_low = bot_rightx_current - margin
            win_xright_high = bot_rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.insert(0, good_left_inds)
            right_lane_inds.insert(0, good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
            if len(good_left_inds) > minpix:
                bot_leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                left_center_line.insert(0, bot_leftx_current)
            else:
                left_center_line.insert(0, None)
            if len(good_right_inds) > minpix:
                bot_rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                right_center_line.insert(0, bot_rightx_current)
            else:
                right_center_line.insert(0, None)
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    else:
        pred_left_fit = left_lane.best_fit
        pred_right_fit = right_lane.best_fit
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        pred_center = pred_left_fit[0] * \
            (nonzeroy ** 2) + pred_left_fit[1] * nonzeroy + pred_left_fit[2]
        left_lane_boarder0 = pred_center - margin
        left_lane_boarder1 = pred_center + margin
        left_lane_inds = ((nonzerox > left_lane_boarder0) &
                          (nonzerox < left_lane_boarder1))

        pred_center = pred_right_fit[0] * \
            (nonzeroy ** 2) + pred_right_fit[1] * nonzeroy + pred_right_fit[2]
        right_lane_boarder0 = pred_center - margin
        right_lane_boarder1 = pred_center + margin
        right_lane_inds = ((nonzerox > right_lane_boarder0)
                           & (nonzerox < right_lane_boarder1))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    pred_left_fit = np.polyfit(lefty, leftx, 2)
    pred_right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = pred_left_fit[0] * ploty ** 2 + \
        pred_left_fit[1] * ploty + pred_left_fit[2]
    right_fitx = pred_right_fit[0] * ploty ** 2 + \
        pred_right_fit[1] * ploty + pred_right_fit[2]

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
    l_points = np.squeeze(np.array(np.dstack((left_x, ploty)), dtype='int32'))
    r_points = np.squeeze(np.array(np.dstack((right_x, ploty)), dtype='int32'))
    out_img[l_points[:, 1], l_points[:, 0]] = [0, 255, 255]
    out_img[r_points[:, 1], r_points[:, 0]] = [0, 255, 255]

    # Draw the search box
    if left_lane.detected == False:
        draw_search_box(left_center_line, right_center_line,
                        image.shape[0], nwindows, margin, out_img)

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
    # print("save lane image: ", savename)
    save_image(out_img, savename, "lane")

    left_lane.update(left_fitx, ploty)
    right_lane.update(right_fitx, ploty)

    return pred_left_fit, pred_right_fit


def draw_search_box(left_search_center, right_search_center, height, num_widows, margin, out_img):
    # Draw the search box
    window_height = int(height / num_widows)
    for n in range(num_widows):
        if left_search_center[n] != None:
            win_x0 = left_search_center[n] - margin
            win_x1 = left_search_center[n] + margin
            win_y0 = height - (n + 1) * window_height
            win_y1 = height - n * window_height
            cv2.rectangle(out_img, (win_x0, win_y0),
                          (win_x1, win_y1), (0, 255, 255))
        if right_search_center[n] != None:
            win_x0 = right_search_center[n] - margin
            win_x1 = right_search_center[n] + margin
            win_y0 = height - (n + 1) * window_height
            win_y1 = height - n * window_height
            cv2.rectangle(out_img, (win_x0, win_y0),
                          (win_x1, win_y1), (0, 255, 255))


def gaussian_fit_points(bot_gaussian_fit, top_gaussian_fit, midpoint, left=True):
    top_mean = top_gaussian_fit[0]
    top_stdev = top_gaussian_fit[1]
    top_max = top_gaussian_fit[2]

    bot_mean = bot_gaussian_fit[0]
    bot_stdev = bot_gaussian_fit[1]
    bot_max = bot_gaussian_fit[2]

    top_candidate = 0.
    if left == True:
        good_position = top_mean < (midpoint * 3 / 4)
    else:
        good_position = top_mean > (midpoint + midpoint / 4)

    if top_stdev < 100 and top_max > 0.1 and good_position:
        top_candidate = top_mean

    bot_candidate = 0.
    if left == True:
        good_position = bot_mean < (midpoint * 3 / 4)
    else:
        good_position = bot_mean > (midpoint + midpoint / 4)

    if bot_stdev < 100 and bot_max > 0.1 and good_position:
        bot_candidate = bot_mean

    if top_candidate == 0 and bot_candidate == 0:
        point = int((top_mean + bot_mean) / 2)
    elif top_candidate != 0 and bot_candidate != 0:
        if (top_max > (2 * bot_max)):
            point = top_candidate
        elif(bot_max > (2 * top_max)):
            point = bot_candidate
        else:
            point = int((top_candidate + bot_candidate) / 2)
    else:
        point = top_candidate + bot_candidate
    return int(point)


def gen_from_hist(bot, top):
    midpoint = np.int(bot.shape[0] / 2)
    top_left = np.copy(top)
    top_left[midpoint:] = 0
    top_right = np.copy(top)
    top_right[:midpoint] = 0

    top_left_fit = gaussian_sfit(top_left)
    top_right_fit = gaussian_sfit(top_right)

    bot_left = np.copy(bot)
    bot_left[midpoint:] = 0
    bot_right = np.copy(bot)
    bot_right[:midpoint] = 0

    bot_left_fit = gaussian_sfit(bot_left)
    bot_right_fit = gaussian_sfit(bot_right)

    left = gaussian_fit_points(bot_left_fit, top_left_fit, midpoint, left=True)
    right = gaussian_fit_points(
        bot_right_fit, top_right_fit, midpoint, left=False)

    return left, right


# Define conversions in x and y from pixels space to meters
IMG_WIDTH = 1280
IMG_HEIGHT = 720
LANE_WIDTH_PX = 700
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


def gen_fit_line(img, lfit, rfit):
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


def plot_fit_line(img, left_plotx, right_plotx, ploty):
    left_points = np.squeeze(
        np.array(np.dstack((left_plotx, ploty)), dtype='int32'))
    right_points = np.squeeze(
        np.array(np.dstack((right_plotx, ploty)), dtype='int32'))

    cv2.polylines(img, [left_points], False, (0, 255, 255), 1)
    cv2.polylines(img, [right_points], False, (0, 255, 255), 1)


def process_image(img, lanes=None, SAVE=""):
    if SAVE == "" and left_lane.savepath != "":
        savename = os.path.join(left_lane.savepath, left_lane.savename)
    else:
        savename = SAVE

    mtx, dist = cam_calib(SAVE=savename)

    # print("process :", fname)
    bin_img = gen_binary_images(img, mtx, dist, savename)
    #print("bin image shape", bin_img.shape, "type", bin_img.dtype)

    warp_img = persp_trans_forward(bin_img)
    #index = np.copy(warp_img)
    #warp_img[index[:,:] > 0] = 255
    #print("warp image shape", warp_img.shape, "type", warp_img.dtype)
    save_image(warp_img, savename, "warp")

    lanes = detect_lanes(warp_img, prev_lanes=None, save_path=savename)

    left_fitx, ploty, right_fitx = gen_fit_line(warp_img, lanes[0], lanes[1])

    out_img = plot_lane(img, left_fitx, ploty, right_fitx)

    # Distance from center
    dist_x = dist_from_center(left_fitx, right_fitx)
    # Radius of curvature
    curverad = get_curverad(ploty, left_fitx, right_fitx)

    # Draw lane into original image, first do inverse perspective tranformation
    out_img = persp_trans_backward(out_img)
    out_img = cv2.addWeighted(img, .5, out_img, .5, 0.0, dtype=0)

    cv2.putText(out_img, "Radius: %.2fm" % curverad, (400, 650),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    if dist_x > 0:
        cv2.putText(out_img, "Right from center: %.2fm" %
                    (np.abs(dist_x)), (400, 700), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    elif dist_x < 0:
        cv2.putText(out_img, "Left from center: %.2fm" %
                    (np.abs(dist_x)), (400, 700), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    else:
        cv2.putText(out_img, "Center", (400, 700),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))

    # print("save name :", save_name)
    save_image(out_img, savename, "final")

    return out_img


def process_video_image(img, lanes=None, SAVE=""):
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    img = process_image(img, lanes=lanes, SAVE=SAVE)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img


def process_test_images():
    images = glob.glob('test_images/*.jpg')
    save_path = "output_images"
    # left_lane = Line()
    # right_lane = Line()

    for fname in images:
        # print("process image:", fname)
        global left_lane
        global right_lane
        left_lane = Lane()
        right_lane = Lane()
        img = cv2.imread(fname)
        head, tail = os.path.split(fname)
        save_name = os.path.join(save_path, tail)
        prev_lanes = None
        process_image(img, lanes=None, SAVE=save_name)


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


def process_video():
    input_video = "project_video.mp4"
    output_video = "project_lane.mp4"

    clip1 = VideoFileClip(input_video)
    output_clip = clip1.fl_image(process_video_image)
    output_clip.write_videofile(output_video, audio=False)


left_lane = Lane()
right_lane = Lane()

#process_test_images()

def main(argv):
    do_process_image = False
    do_process_video = False
    debug = False

    while(len(argv) != 0):
        command = argv.pop()

        if command == "image":
            do_process_image = True
        elif command == "video":
            do_process_video = True
        elif command == "debug":
            debug = True

    if do_process_image:
        print("Processing test images...")
        process_test_images()

    if do_process_video:
        print("Processing video...")
        if debug == True:
            global left_lane
            global right_lane
            left_lane = Lane(os.path.join(TMP_DIR, "video"))
            right_lane = Lane(os.path.join(TMP_DIR, "video"))
        process_video()

    pass

if __name__ == "__main__":
    main(sys.argv)
