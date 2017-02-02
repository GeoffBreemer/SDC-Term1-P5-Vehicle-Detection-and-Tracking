'''
Extracts a number of features from images to 1) train a VehicleClassifier and b) use in a VehicleDetector to predict
vehicles in a sequence of sliding windows. Features used: HOG, spatial binning and histogram of colors
'''
import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog

class FeatureExtractor():
    def __get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis, feature_vec):
        '''Calculate HOG features and visualization'''

        if vis == True:
            return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        else:
            return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)


    def __bin_spatial(self, img, size):
        '''Calculate binned color features'''
        return cv2.resize(img, size).ravel()


    def __color_hist(self, img, nbins, bins_range):
        '''Calculate histograms, bin centers and feature vector'''
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

        # Return the individual histograms, bin_centers and feature vector
        return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))


    def extract_features_single_image(self, image, cspace, spatial_size, hist_bins, hist_range,
                                      hog_orient, hog_pix_per_cell, hog_cell_per_block, hog_channel):
        '''Extract features for a single image'''
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        else:
            feature_image = np.copy(image)

        # Apply __bin_spatial() to get spatial color features
        spatial_features = self.__bin_spatial(feature_image, size=spatial_size)

        # Apply __color_hist() also with a color space option now
        hist_features = self.__color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Extract HOG features
        if hog_channel == 'ALL':
            hog_features = []
            # Process each color channel
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.__get_hog_features(feature_image[:, :, channel],
                                                            hog_orient, hog_pix_per_cell, hog_cell_per_block,
                                                            vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            # Use a 3 channel color space
            hog_features = self.__get_hog_features(feature_image[:, :, hog_channel], hog_orient,
                                                   hog_pix_per_cell, hog_cell_per_block, vis=False, feature_vec=True)

        # Return the feature vectors
        return spatial_features, hist_features, hog_features


    def extract_features(self, image_names, cspace, spatial_size, hist_bins, hist_range,
                         hog_orient, hog_pix_per_cell, hog_cell_per_block, hog_channel):
        '''Extract features for a list of images'''

        features = []

        # Iterate through the list of images and extract features
        for file in image_names:
            image = mpimg.imread(file)              # PNG: 0-1, JPG: 0-255
            image = np.uint8(image * 255)           # Scale training images in PNG from 0-1 to 0-255

            spatial_feat, hist_feat, hog_feat = self.extract_features_single_image(image, cspace, spatial_size,
                                                                                   hist_bins, hist_range,
                                                                                   hog_orient, hog_pix_per_cell,
                                                                                   hog_cell_per_block, hog_channel)

            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_feat, hist_feat, hog_feat)))

        # Return list of feature vectors
        return features