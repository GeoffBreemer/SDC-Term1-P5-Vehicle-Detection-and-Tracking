'''
Classifier that classifies images as vehicles or non-vehicles. Uses a FeatureExtractor to extract features from the
training images
'''
import numpy as np
import glob, time
from FeatureExtractor import FeatureExtractor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from Config import *

# Miscellaneous constants
RANDOM_STATE = 122177
TEST_TRAIN_RATIO = 0.1              # Portion of training images to be used as a test set

class VehicleClassifier():
    '''Classifier that detects vehicles in an image'''
    def __init__(self, trained_model_path=None):
        '''
        Initialise the object
        :param trained_model_path: full path to were the trained model and scaler will be stored
        '''
        self.trained_model_path = trained_model_path
        self.clf = None
        self.X_scaler = None
        self.trained = False
        self.feature_extractor = None


    def load_training_images(self, vehicle_path=None, non_vehicle_path=None):
        '''
        Load training set image names from disc

        :param vehicle_path: path to where the vehicle training images are stored
        :param non_vehicle_path: path to where the non-vehicle training images are stored
        :return: two lists with full paths to vehicle and non-vehicle images respectively
        '''
        vehicles = []
        non_vehicles = []

        # Vehicle images names
        print("Loading training image names...")
        for image in glob.glob(vehicle_path + '/**/*.png', recursive=True):
            vehicles.append(image)

        # Non-vehicle images names
        for image in glob.glob(non_vehicle_path + '/**/*.png', recursive=True):
            non_vehicles.append(image)

        print('    # of vehicle images: {}'.format(len(vehicles)))
        print('# of non-vehicle images: {}'.format(len(non_vehicles)))

        return vehicles, non_vehicles


    def extract_features(self, vehicles, non_vehicles):
        '''
        Extract features for the two lists containing vehicle and non-vehicle image paths respectively
        :param vehicles: list of paths to vehicle images
        :param non_vehicles: list of paths to non-vehicle images
        :return: scaled_X: normalised feature vector, y: true labels (1 = vehicle, 0 = non-vehicle)
        '''
        '''Load training set images and extract features'''
        self.feature_extractor = FeatureExtractor()

        print("Loading images and extracting features...")
        t = time.time()
        vehicle_features = self.feature_extractor.extract_features(vehicles,
                                                                   cspace=CSPACE,
                                                                   spatial_size=(SPATIAL_SIZE, SPATIAL_SIZE),
                                                                   hist_bins=HIST_BIN, hist_range=HIST_RANGE,
                                                                   hog_cell_per_block=HOG_CELL_PER_BLOCK,
                                                                   hog_channel=HOG_CHANNEL,
                                                                   hog_pix_per_cell=HOG_PIX_PER_CELL,
                                                                   hog_orient=HOG_ORIENT_BINS)

        non_vehicle_features = self.feature_extractor.extract_features(non_vehicles,
                                                                       cspace=CSPACE,
                                                                       spatial_size=(SPATIAL_SIZE, SPATIAL_SIZE),
                                                                       hist_bins=HIST_BIN,
                                                                       hist_range=HIST_RANGE,
                                                                       hog_cell_per_block=HOG_CELL_PER_BLOCK,
                                                                       hog_channel=HOG_CHANNEL,
                                                                       hog_pix_per_cell=HOG_PIX_PER_CELL,
                                                                       hog_orient=HOG_ORIENT_BINS)

        # Create an array stack of all feature vectors and scale the resulting feature vector
        X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
        self.X_scaler = StandardScaler().fit(X)
        scaled_X = self.X_scaler.transform(X)

        # Define the labels vector (1 = vehicle, 0 = non-vehicle)
        y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
        t2 = time.time()

        print('Number of features: {}'.format(scaled_X.shape[1]))
        print('Feature extraction time: {}'.format(round(t2 - t, 2)))

        return scaled_X, y


    def train(self, X, y):
        '''
        Train the classifier using training set X and true labels y

        :param X: array of feature vectors (normalised)
        :param y: array of true predictions
        '''

        # Split the training set into randomized training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_TRAIN_RATIO,
                                                            random_state=RANDOM_STATE, stratify=y)

        # Train the linear SVC
        print("Training SVC...")
        self.clf = LinearSVC()
        t = time.time()
        self.clf.fit(X_train, y_train)
        t2 = time.time()
        print('Classifier training time: {}'.format(round(t2 - t, 2)))

        self.trained = True

        # Check the score of the SVC on the test set
        print('Classifier test set accuracy: {}'.format(round(self.score(X_test, y_test), 4)))


    def save_model(self, name):
        '''
        Save the trained model and scaler to disc

        :param name: name of the model ("_model.pkl" will be added to the name)
        '''
        if self.trained:
            joblib.dump(self.clf, self.trained_model_path + '/' + name + '_model.pkl')
            joblib.dump(self.X_scaler, self.trained_model_path + '/' + name + '_scaler.pkl')
        else:
            print("ERROR: model not yet trained")


    def load_model(self, name):
        '''
        Load a trained model from disc

        :param name: name of the model ("_model.pkl" will be added to the name)
        '''
        self.__init__(self.trained_model_path)

        # Load the trained classifier and the scaler
        self.clf = joblib.load(self.trained_model_path + '/' + name + '_model.pkl')
        self.X_scaler = joblib.load(self.trained_model_path + '/' + name + '_scaler.pkl')
        self.feature_extractor = FeatureExtractor()

        self.trained = True


    def predict(self, X):
        '''
        Make predictions for test set in X

        :param X: array of feature vectors
        :return: array of predictions
        '''
        if self.trained:
            return self.clf.predict(X)
        else:
            print("ERROR: model not yet trained")


    def score(self, X, y):
        '''
        Determine the classifier accuracy given test set X and true labels y

        :param X: array of feature vectors (normalised)
        :param y: array of predictions
        :return: classifier accuracy
        '''
        if self.trained:
            return self.clf.score(X, y)
        else:
            print("ERROR: model not yet trained")
