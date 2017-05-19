**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Original"
[image2]: ./output_images/calibration2_chess.jpg "Chess Board"
[image3]: ./output_images/calibration2_undistort.jpg "Undistort Image"
[image4]: ./output_images/straight_lines1_undistort.jpg "Undistort Test Images"
[image5]: ./output_images/undistort/straight_lines1.jpg "Test Image 1"
[image6]: ./output_images/undistort/00001043.jpg "Test Image 2"
[image7]: ./output_images/00001043_binary.jpg "Test binary"
[image8]: ./output_images/combined/straight_lines1.jpg "Binary 1"
[image9]: ./output_images/combined/00001043.jpg "Binary 2"
[image10]: ./output_images/straight_lines1_perspective.jpg "Perspective 1"
[image11]: ./output_images/00001043_perspective.jpg "Perspective 2"
[image12]: ./output_images/warp/straight_lines1.jpg "Warp Example 1"
[image13]: ./output_images/warp/00001043.jpg "Warp Example 2"
[image14]: ./output_images/bottom_hist/straight_lines1.jpg "Bottom Gaussian Fit"
[image15]: ./output_images/top_hist/straight_lines1.jpg "Top Gaussian Fit"
[image16]: ./output_images/lane/straight_lines1.jpg "Searching Window"
[image17]: ./output_images/final/straight_lines1.jpg "Final 1"
[image18]: ./output_images/final/00001043.jpg "Final 2"
[video1]: ./project_lane.mp4 "Video"

---
# 1. Camera Calibration

The code for this step is in *do_camera_calibration* function from lines 12 through 71 of the file `camera_calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]
![alt text][image3]

After do and check the result, I then save the calibration matrix into pickle file that I can load it then.

And while apply camera distortion to test images, one example is here:

![alt text][image4]

# 2. Binary Image Creation

To demonstrate my processing pipeline, I will describe how I apply the distortion correction to one of the test images like these two, one from test image sets and one from test video frame:

![alt text][image5]

![alt text][image6]

I used a combination of color, luminance and gradient thresholds to generate a binary image. First, I translate the distorted image from RGB to HLS and grayscale image; Then I apply Sobel edge detection on grayscale image and extract S channel from HLS image as it represents color information, and so as the L channel for lightness information.

At first I was using only gradient and S values only, but I found that, in video frames, there are shadows on the road, which also have large S values since they are 'Saturated'. Use L channel, which is lightness, I think, could mask out these dark shadows from final result. Here's an example:

![alt text][image7]

First row in the middle is the original image, right to it is the result of Sobel operation. 

First image in the second row is the S channel of the original image, second image is the L channel of the original image and third image is applied threshold to the Sobel image.

In the third row, first one is the thesholded S channel, and so forth the second image for the L channel, the third image is S & L, which means only where the S channel value in between threshold and L channel value larger than the threshold points are left.

At the fourth row, first image is S | Sobel, the second is (S & L) | Sobel. It is very clear that large shadows on the road which have high S value are masked out in (S & L) | Sobel image.

The thresholding steps at lines 40 through 74 in `Advanced_Line_Lines.py`).  Finally, the result binary image are like this:

![alt text][image8]

![alt text][image9]

# 3. Perspective Transform

The code for my perspective transform appears in lines 10 through 33 in the file `perspective.py`.  The `persp_trans_forward` function takes as inputs an image (`img`) and transform it, using global variable (`src`) as source point and global variable (`dst`) as destination points. And the `persp_trans_backword` function do the reverse as `persp_trans_forward` do. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[545, 480], [742, 480], [1030, 660], [280, 660]])
dst = np.float32([[220, 100], [1060, 100], [1060, 700], [220, 700]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 545, 480      | 220, 100      | 
| 742, 480      | 1060, 100     |
| 1030, 660     | 1060, 700     |
| 280, 660      | 220, 700      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]

![alt text][image11]

Apply perspective transform define above, I got:

![alt text][image12]

![alt text][image13]

# 4. Find Lane Lines
For single image or frames in video which didn't find lane lines before, I use sliding window serch and fit lane lines with 2nd order polynomial, for video frames, I use class `Lane` to remember prior n fits and average them as a prediction of current frame and fit lane lines then.

## 4. 1 Sliding Window Search
First, I apply histogram of top and bottom half to the warped binary image, code are in `Advanced_Lane_Lines.py`.
```python
bot_histogram = np.sum(image[int(image.shape[0] / 2):, :], axis=0)
top_histogram = np.sum(image[:int(image.shape[0] / 2), :], axis=0)
```

Then, in function `gen_from_hist`(`Advanced_Lane_Lines.py`, lines 451 to 473), I first separate top histogram to left and right part, and so as to bottom histogram(`Advanced_Lane_Lines.py`, line 452-456), then do gaussian fit to the histogram with peak as the mean of gaussian fit, this is done in function `gaussian_sfit`(`gaussian_fit.py`, line 85-90). Thus, I get four gaussian fitted lines for top left, top right, bottom left, bottom right:
![alt text][image14]
![alt text][image15]

Then I choose top and bottom fit points according to the fit parameters in function `gaussian_fit_points`(`Advanced_Lane_Lines.py`, 410-448). After left and right lane starting points are set, I use sliding window to find lane points, codes are in function `detect_lanes`(`Advanced_Lane_Lines.py`, 169-299). Currently I use 9 windows, margin as 100, and if minimum 50 pixels found to recenter the searching window. I start searching from the center height of the image and to both top and bottom side step by step.

Then I use found lane point to fit with a 2nd order polynomial.
```python
# File: Advanced_Lane_Lines.py Line: 330-331
pred_left_fit = np.polyfit(lefty, leftx, 2)
pred_right_fit = np.polyfit(righty, rightx, 2)
```
Searching window and fitted line are like this:

![alt text][image16]

## 4. 2 Previous Found Lines as prediction
When processing video, I use previous frame line as a prediction for current frame. I use a class `Lane`(`Advanced_Lane_Lines.py`:97-153) to save n frame fitted line parameters. It works like this:
1. use average n(3 frames right now) fits as prediction (`Advanced_Lane_Lines.py`: 302-303)
2. use predicted lines to find lane pixels, and do a 2nd order fit to these points to find the actual lane center (`Advanced_Lane_Lines.py`: 309-338)
3. update and save current line into class `Lane` (`Advanced_Lane_Lines.py`: 384-385)
    - append current line into previous lines (`Advanced_Lane_Lines.py`: 144-145)
    - if saved lines are more than n, discard the oldest one (`Advanced_Lane_Lines.py`: 149)
    - average all saved lines to create a predicted one

## 5. Calculate the radius of curvature and vehicle departion to center

I calculate the radius of curvature according to the quation in project class part 35, this is done in fuction `get_curverad` in file `Advanced_Lane_Lines.py`:498-508.

I also calculate the offset from center through function `dist_from_center`(`Advanced_Lane_Lines.py`:486-493), according to project class part 36, just convert the distance between image center and lane center with pixels to meters. Following predefined parameters I just adopt from class, and it works correctly according to the result.
```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

## 6. Plot result back to road

I implemented this step in lines 570 through 584 in my code in `Advanced_Lane_Lines.py` in the function `process_image()`.

Here is an example of my result on a test image:

![alt text][image18]

---

# 7. Video result

Video processing is almost the same as image pipeline, except that when processing video, previous lines are important to make prediction more efficient and robust.

Video reading are in function `process_video()` in file `Advanced_Lane_Lines.py`:634-640. Because I use cv2.imread() in whole pipeline, so I have to do color convert from RGB to BGR before processing and then convert it back to RGB after processing, function `process_video_image()` in file `Advanced_Lan_Lines.py`:692-698 help did this.

Here's a [link to my video result](./project_video.mp4)

---

## 8. Discussion

The most problem I encountered is how to tackle the shadows on the road, which affects the lane finding performance on video frames a lot. My try to tune the parameter seems to not help much, then I find lightness could help to improve this and it works fine.

According to the whole pipeline, I can still think that shadows, high lights, water reflections are certan source that may make pipeline fail. We also suppose that the road is a plane, so that is why perspective transform works, if the road is up or down, that would likely to make pipeline fail.

To make this pipeline more robust, I think maybe need a more robust line finding method, maybe cnn could have help. And if we can get more information from hardware and car, such as camera inner/outter parameters, the car's movements, acceleration, rotation, etc., together with video, would make it more robust.

