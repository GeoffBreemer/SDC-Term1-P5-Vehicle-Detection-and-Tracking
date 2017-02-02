'''
Detect vehicles in an image or video frame using a trained VehicleClassifier
'''
import numpy as np
import cv2
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
from Config import *

# Miscellaneous constants
FIG_SHAPE = (12, 8)
TRAINING_IMAGE_SHAPE = (64, 64)

# Detection constants (changes DO NOT require VehicleClassifier retraining)
HEAT_THRESHOLD = 3                  # threshold for a single frame's heatmap
HEAT_SMOOTH_FACTOR = 0.0            # smoothing factor for heatmaps of the current and previous frame
HEAT_MULTI_FRAME_THRESHOLD = 8      # threshold for the heatmap of a number of frames combined

# Debugging helper functions
def plot_diagnostics(image_final, heat_current,
                     heat_current_threshold,
                     heat_combined,
                     heat_combined_threshold):
    '''Plot a number of images created by the pipeline into a single image'''
    diagScreen = np.zeros((1080, 1280, 3), dtype=np.uint8)

    # Main screen
    diagScreen[0:720, 0:1280] = image_final

    # Four screens along the bottom
    heat_current = np.dstack((heat_current, heat_current, heat_current)) * 255
    diagScreen[720:1080, 0:320] = cv2.resize(heat_current, (320, 360), interpolation=cv2.INTER_AREA)

    heat_current_threshold = np.dstack((heat_current_threshold, heat_current_threshold, heat_current_threshold)) * 255
    diagScreen[720:1080, 320:640] = cv2.resize(heat_current_threshold, (320, 360), interpolation=cv2.INTER_AREA)

    heat_combined = np.dstack((heat_combined, heat_combined, heat_combined)) * 255
    diagScreen[720:1080, 640:960] = cv2.resize(heat_combined, (320, 360), interpolation=cv2.INTER_AREA)

    heat_combined_threshold = np.dstack((heat_combined_threshold, heat_combined_threshold, heat_combined_threshold)) * 255
    diagScreen[720:1080, 960:1280] = cv2.resize(heat_combined_threshold, (320, 360), interpolation=cv2.INTER_AREA)

    return diagScreen


# Detection helper functions
def draw_boxes(img, bboxes, color, thick):
    '''Draw one or more bounding boxes in an image'''
    imcopy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


def slide_window(img, x_start_stop, y_start_stop, xy_window, xy_overlap):
    '''Create a list of windows covering the image'''
    # If x and/or y start/stop positions not defined, set them to the image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0

    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]

    if y_start_stop[0] == None:
        y_start_stop[0] = 0

    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Append window position to list
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list


def add_heat(heatmap, bbox_list):
    '''Add "heat" to a map for a list of bounding boxes'''
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    '''Zero out heatmap pixels below the threshold and return thresholded heatmap'''
    new_heatmap = np.copy(heatmap)
    new_heatmap[new_heatmap <= threshold] = 0

    return new_heatmap


def draw_labeled_bboxes(image, labels):
    '''Draw rectangles around labeled regions'''
    # Iterate through all detected vehicles
    for vehicle_id in range(1, labels[1] + 1):
        # Find pixels with each vehicle_id label value
        nonzero = (labels[0] == vehicle_id).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)

    # Return the image
    return image


class VehicleDetector():
    '''Classifier that detects vehicles in an image'''
    def __init__(self, classifier, output_path=None):
        '''
        Initialise the detector
        :param classifier: trained VehicleClassifier
        :param output_path: path where images are to be save, only relevant when using verbose = True in detect()
        '''
        self.clf = classifier
        self.previous_heatmap = None
        self.output_path = output_path


    def __search_windows(self, img, windows):
        '''
        Search each sliding window for images of vehicles using the classifier
        :param img: the original image/frame
        :param windows: list of sliding windows
        :return: list of windows for which the classifier predicts the window contains a vehicle
        '''
        on_windows = []

        # Iterate over all windows in the list
        for window in windows:
            # Extract the test window from the original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], TRAINING_IMAGE_SHAPE)

            # Extract features for that window
            spatial_features, hist_features, hog_features =\
                self.clf.feature_extractor.extract_features_single_image(test_img,
                                                                     cspace=CSPACE,
                                                                     spatial_size=(SPATIAL_SIZE, SPATIAL_SIZE),
                                                                     hist_bins=HIST_BIN,
                                                                     hist_range=HIST_RANGE,
                                                                     hog_cell_per_block=HOG_CELL_PER_BLOCK,
                                                                     hog_channel=HOG_CHANNEL,
                                                                     hog_pix_per_cell=HOG_PIX_PER_CELL,
                                                                     hog_orient=HOG_ORIENT_BINS)

            features = np.concatenate((spatial_features, hist_features, hog_features))

            # Scale extracted features to be fed to the classifier
            test_features = self.clf.X_scaler.transform(np.array(features).reshape(1, -1))

            # Predict using the classifier
            prediction = self.clf.predict(test_features)

            # If prediction is 1 (= vehicle) then save the window
            if prediction == 1:
                on_windows.append(window)

        # Return windows for positive detections
        return on_windows


    def detect(self, frame, verbose=False, new_image=False):
        '''
        Perform vehicle detection.

        :param frame: a single image or video frame, must be an RGB with pixel values 0-255
        :param verbose: set to True to show and save to disc various stages of the function (do not use with videos)
        :param new_image: set to True if the detector is being used on unrelated images, set to False when processing
        video frames
        :return: original image/frame with bounding boxes that identify vehicles
        '''
        org_image = np.copy(frame)
        all_windows = []

        # If the same detector is used on a sequence of individual but unrelated images, as opposed to subsequent
        # frames of the same video, the previous_heatmap needs to be reset for every call. For video frames processed
        # in succession this parameter should be set to False
        if new_image:
            self.previous_heatmap = None

        # Grid 1 (closest to the horizon)
        win_size = 60
        y_start = int(frame.shape[0] / 2) + 20
        y_stop = y_start + win_size / 0.75
        x_start = 480
        x_stop = None
        windows_small = slide_window(frame, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop],
                                     xy_window=(win_size, win_size), xy_overlap=(0.75, 0.75))
        all_windows += windows_small

        if verbose:
            window_img = draw_boxes(frame, windows_small, color=(0, 0, 255), thick=2)

        # Grid 2
        win_size = 80
        y_start = int(frame.shape[0] / 2) + 40
        y_stop = y_start + win_size / 0.75
        x_start = 400
        x_stop = None
        windows_med = slide_window(frame, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop],
                                   xy_window=(win_size, win_size), xy_overlap=(0.75, 0.75))
        all_windows += windows_med

        if verbose:
            window_img = draw_boxes(window_img, windows_med, color=(0, 255, 0), thick=2)

        # Grid 3
        win_size = 120
        y_start = int(frame.shape[0] / 2) + 60
        y_stop = y_start + win_size / 0.75
        x_start = 380
        x_stop = None
        windows_bot = slide_window(frame, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop],
                                   xy_window=(win_size, win_size), xy_overlap=(0.75, 0.75))

        all_windows += windows_bot
        if verbose:
            window_img = draw_boxes(window_img, windows_bot, color=(255, 0, 0), thick=2)

        # Grid 4 (closest to the camera car)
        win_size = 160
        y_start = int(frame.shape[0] / 2) + 80
        y_stop = y_start + win_size / 0.75 - 60
        x_start = 340
        x_stop = None
        windows_bot = slide_window(frame, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop],
                                   xy_window=(win_size, win_size), xy_overlap=(0.75, 0.75))

        all_windows += windows_bot
        if verbose:
            window_img = draw_boxes(window_img, windows_bot, color=(255, 255, 0), thick=2)


        if verbose:
            print('Number of sliding windows used: {}'.format(len(all_windows)))

        # 2. Extract features for each sliding window and predict whther it contains a vehicle using the classifier
        hot_windows = self.__search_windows(org_image, all_windows)

        # Draw all windows that contain a detected vehicle (may include overlaps and false positives)
        if verbose:
            window_det = draw_boxes(org_image, hot_windows, color=(255, 255, 255), thick=6)
            plt.figure(figsize=FIG_SHAPE)
            plt.imshow(window_det)
            plt.show()
            image_tmp = cv2.resize(window_det, (int(window_det.shape[1] / 2), int(window_det.shape[0] / 2)))
            plt.imsave(self.output_path + '/all_detections.jpg', image_tmp)

        # 3. Combine duplicate detections by creating a heatmap
        current_heatmap = np.zeros_like(frame[:, :, 0]).astype(np.float)
        current_heatmap = add_heat(current_heatmap, hot_windows)

        # Show heatmap prior to thresholding
        if verbose:
            plt.figure(figsize=FIG_SHAPE)
            plt.imshow(current_heatmap, cmap='hot')
            plt.show()
            image_tmp = cv2.resize(current_heatmap, (int(current_heatmap.shape[1] / 2), int(current_heatmap.shape[0] /
                                                                                          2)))
            plt.imsave(self.output_path + '/heatmap_prior_thresholding.jpg', image_tmp)

        # 4. Threshold the heatmap to remove false positives and duplicate detections
        current_heatmap_thresh = apply_threshold(current_heatmap, HEAT_THRESHOLD)

        # Show heatmap prior to smoothing
        if verbose:
            plt.figure(figsize=FIG_SHAPE)
            plt.imshow(current_heatmap_thresh, cmap='hot')
            plt.show()
            image_tmp = cv2.resize(current_heatmap_thresh, (int(current_heatmap_thresh.shape[1] / 2),
                                                            int(current_heatmap_thresh.shape[0] / 2)))
            plt.imsave(self.output_path + '/heatmap_post_thresholding.jpg', image_tmp)

        # 5. Determine the number of vehicles and their position by identifying the positions and regions in the heatmap
        if self.previous_heatmap is None:
            # There is no previous frame heat map so just use blank images
            current_heatmap_combined = np.zeros_like(frame[:, :, 0]).astype(np.float)
            current_heatmap_combined_thresh = current_heatmap_combined

            labels = label(current_heatmap_thresh)
            self.previous_heatmap = current_heatmap_thresh
        else:
            # Use a smoothing factor to combine the current and previous frame heat map
            current_heatmap_combined = self.previous_heatmap * HEAT_SMOOTH_FACTOR +\
                                       current_heatmap_thresh * (1 - HEAT_SMOOTH_FACTOR)

            # Apply a different threshold to the combined heatmap
            current_heatmap_combined_thresh = apply_threshold(current_heatmap_combined, HEAT_MULTI_FRAME_THRESHOLD)

            labels = label(current_heatmap_combined_thresh)
            self.previous_heatmap = current_heatmap_combined_thresh

        # 6. Draw the bounding boxes of the detected regions in the original image/frame
        window_hot = draw_labeled_bboxes(np.copy(frame), labels)

        # Show detected car blobs
        if verbose:
            plt.figure(figsize=FIG_SHAPE)
            plt.imshow(labels[0], cmap='gray')
            plt.show()
            plt.imsave(self.output_path + '/labeled_regions.jpg', labels[0])

        # Create a diagnostic view
        # if verbose:
        #     image_diag = plot_diagnostics(window_hot, current_heatmap, current_heatmap_thresh,
        #                               current_heatmap_combined, current_heatmap_combined_thresh)
        #     image_diag = cv2.resize(image_diag, (int(image_diag.shape[1] / 2), int(image_diag.shape[0] / 2)))
        #     plt.imsave(self.output_path + '/image_diag.jpg', image_diag)

        if verbose:
            return window_img, all_windows, window_hot, hot_windows
        else:
            return window_hot
