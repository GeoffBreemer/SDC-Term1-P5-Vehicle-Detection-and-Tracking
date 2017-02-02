# Project 5: Vehicle Detection and Tracking
This document describes the submission for **Project 5: Vehicle Detection and Tracking**.

## Table of Contents
- [Project code](#project-code)

- [Running the project](#running-the-project)

- [Pipeline](#pipeline)

- [Project video](#project-video)

- [Discussion and challenges](#discussion-and-challenges)

# Project code
All code for this project is implemented in the following `.py` files:
- `vehicle_detection_and_tracking.py`: this is the main script for this project that ties everything together 

- `FeatureExtractor.py`: class that handles feature extraction

- `VehicleClassifier.py`: class that handles classifier training

- `VehicleDetector.py`: class that handles vehicle detection

- `Config.py`: contains a number of feature extraction constants used by the `VehicleClassifier` and `VehicleDetector` classes.

# Running the project
To run this project simply execute `vehicle_detection_and_tracking.py`. Note that it is currently setup to produce all images included in the `output_images` folder, train the classifier from scratch and process the project video `project_video.mp4`. On available hardware it took about 30 minutes to finish.

# Pipeline
The pipeline for this project comprises the following steps:

1. Load training data

2. Extract features

3. Train classifier

4. Detect vehicles

5. Output visual display of detected vehicles.

Each step is discussed in more detail below, including references to various Python classes and functions implemented for each step.

## Step 1: Load training data
Loading of the training data is implemented in function `load_training_images` of the `VehicleClassifier` class. All training data provided by Udacity (i.e. GTI, KITTI and project video extracts) have been used for this project.

Because the vehicle and non-vehicle data sets contain roughly the same number of images it was deemed unnecessary to ceate additional training images for one of the data sets.

## Step 2: Extract features
Feature extraction is implemented in function `extract_features` of the `FeatureExtractor` class. It combines a number of features into one long one-dimensional feature vector for each training image:

1. **Histogram of Oriented Gradient (HOG)**: HOG features are extracted in function `__get_hog_features` and returned as a one-dimensional feature vector 
2. **Spatial binning of color**: function `__bin_spatial` returns a scaled down version of an image as a one-dimensional feature vector 
3. **Histograms of color**: function `__color_hist` returns the histograms of all three color channels as a single one-dimensional feature vector. 

After extracting features the resulting one-dimensional feature vector is normalized using scikit-learn's `StandardScaler` class. All three feature vectors combined have about a dozen parameters to tune. As the pipeline was being developed, parameter values were set to the default values discussed in class. Once the pipeline worked end-to-end, various values were tried (all parameter values are defined in `Config.py`):

1. **Color space**: based on the lectures it was assumed the RGB color space was not the right color space to use. After some tinkering the `YUV` and `YCrCb` color spaces were experimented with, and the `YCrCb` color space was finally settled on

2. **HOG variables**: a handful of different values for HOG parameters were trialled, but it seems the default ones performed satisfactorily, i.e. `2` cells per block, color channel `0`, `8` pixels per cell and `9` orientation bins

3. **Spatial size**: in an attempt to speed up processing the spatial size was reduced to `(16, 16)` from the default `(32, 32)`

4. **Color histograms**: the range was kept at `(0, 256)` and the number of bins was reduced to `16` from the default `32`.

The resulting feature vector used for this project contains `2,580` features.

## Step 3: Train classifier
A linear Support Vector Machine is used as the classifier. It is trained in function `train` of the `VehicleClassifier` class. Before training commences the training data is split into a large training set (90%) and a small test set (10%).

Classifier accuracy on the test set is `98.54%`. Changing hyperparameters and/or using a different classifier was considered but not implemented because performance seems adequate with just the default parameters. 

After training, both the classifier and feature scaler are saved to disc by the `save_model` function. They can be loaded again using function `load_model`. This means a trained model can be used repeatedly without having to train the model from scratch as the pipeline is developed further.

## Step 4: Detect vehicles
The vehicle detection pipeline is contained in the `detect` function of the `VehicleDetector` class. It performs the following actions:

- Create sliding windows grid

- Extract features and predict

- Create a heatmap

- Eliminate false positives and duplicate detections

- Determine vehicle positions

The `detect` function requires a trained `VehicleClassifier`.

### Create sliding windows grid
First a grid of sliding windows is created. The grid is setup in such a way that areas in the image that are unlikely to contain vehicles are not covered by sliding windows. For example the sky, the left part of the image (implying that vehicles driving on the other side of the barrier will not be detected) and the bottom of the image (which is obstructed by the bonnet of the camera car). This reduces detection time and the number of false positives. In total `570` sliding windows are used for this project.

The sliding windows are also created using different scales. Sliding windows near the horizon are smaller, sliding windows closer to the camera car are larger. A total of four different scales have been used. The resulting sliding window grid, with each of the four scales drawn using a different color, is shown in the image below:

![image1]

### Extract features and predict
For each sliding window the associated features are extracted, normalized using the same scaler used to normalize the training data, and fed to the classifier. This results in a list of sliding windows the classifier believes contain a vehicle

The results for test image `test3.jpg` are shown below:

![image7]

### Create a heatmap
Using the sliding windows for which a vehicle is predicted, a heatmap is created. This is an image where each filled sliding window is plotted on top of each other, increasing pixel intensity every time a window is plotted. More intense pixels imply they were detected by multiple sliding windows.

This results in an image like the one shown below for test image `test3.jpg`:

![image2]

### Eliminate false positives and duplicate detections
By thresholding the heatmap false positives can be filtered. Essentially any detections that are not (partially) covered by a minimum number of sliding windows is discarded. The heatmap also helps combine duplicate detections into a single detection. For this project a heatmap threshold of `3` was used.

For test image `test3.jpg` this results in the thresholded heatmap shown below:

![image3]

### Determine vehicle positions
The final step is to use the heatmap to determine the number of vehicles and, more importantly, their bounding boxes. This is done using `scipy.ndimage.measurements`'s `label` function. This function determines the bounding boxes for all regions it detected in an image.

The regions for test image `test3.jpg` are shown below:

![image4]

## Step 5: Output visual display of detected vehicles
With the regions containing detected vehicles known it is simply a matter of drawing bounding boxes around each region on the image or video frame and displaying the result.

The detected vehicles in test image `test3.jpg` is shown below:

![image5]

## Results for all six test images
The results for each of the six test images are shown below. The code for detecting vehicles in individual images is implemented in `vehicle_detection_and_tracking.py`'s `process_test_images` function. It simpy calls `VehicleDetector`'s `detect` function for each image and then plots the original images and the results in a single image.

![image6]

# Project video
Exactly the same pipeline is applied to videos. The pipeline is called from `vehicle_detection_and_tracking.py`'s `process_video` function. This function also calls `VehicleDetector`'s `detect` function for each frame.

Here is the [link to the project video result](./output_project_video.mp4).

The result contains a handful of false positives. The rectangles that identify vehicles also tend to be a bit jittery. Frame-to-frame 'smoothing' of the heatmaps was attempted, but abandoned due to time constraints. To assist with this a diagnostic view was developed, showing detection results at the top. Along the bottom are shown the:

- current frame's heatmap
- thresholded version of the heatmap
- heatmap for the current and previoud frame's heatmap
- tresholded version of the heatmap

An example of the diagnostic view is shown below:

![image8]

# Discussion and challenges
The key challenge associated with this project was, as often is the case, parameter tuning. Parameter tuning was first conducted on the six test images. After achieving good results on the images, the pipeline was applied to the video. This is were another challenge surfaced: it takes quite a while to process the entire video. To cut down development time, a very short version of only `97` frames, and a slightly longer version of `475` frames was created.

Values for the parameters found while testing the pipeline on the six test images did not perform as well on the video. Further tweaking of the sliding windows grid and heatmap threshold was required to get the pipeline to work properly on both the images and the project video. One of the issues with finding the right parameter settings is that parameters may interact with each other: the best value for one parameter may require a change to another, previously optimized, parameter.

While a significant part of the code was discussed in some form in the lectures, putting it all together to create a working pipeline and finding parameter values that work reasonably well took quite some time.

Future improvements include:

1. Better error handling, which is currently only implemented in limited places (current code is not exactly production ready)

2. Additional feature extraction tweaking, e.g. by further tweaking parameter values using a grid search approach, removing duplicate and highly correlated features, etc.

3. Trialling different classifiers to achieve better classification accuracy

4. Data augmentation (e.g rotating, flipping images) and hard negative mining to help improve the classifier and prevent false positives

5. Improve generalization. The current pipeline has been optimized for the project video, which always sees the car drive on a stretch of road curving slightly right. It may not perform as well, and possibly skip signficant parts of the image if the car changes lanes, drives on different roads etc.

6. Developing a convolutional neural network approach.

[//]: # (Image References)

[image1]: output_images/test_sliding_windows_grid.jpg "Sliding window grid"
[image2]: output_images/heatmap_prior_thresholding.jpg "Heatmap prior to thresholding"
[image3]: output_images/heatmap_post_thresholding.jpg "Heatmap post thresholding"
[image4]: output_images/labeled_regions.jpg "Labeled detections"
[image5]: output_images/test_hot_windows.jpg "Final detections"
[image6]: output_images/test_detected_all6.png "Detected vehicles for al six test images"
[image7]: output_images/all_detections.jpg "All detections prior to removing false positives and duplicate detections"
[image8]: output_images/image_diag.jpg "Diagnostic view"
[video1]: project_video.mp4 "Project video"

