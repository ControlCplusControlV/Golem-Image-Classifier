#!/usr/local/bin/python
import numpy as np
import scipy
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import os
import datetime
import tarfile
import urllib.request
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import time
import numpy as np
import pickle
import os
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import joblib
import argparse
warnings.filterwarnings('ignore')
#Define an arg parsers to build out a CLI
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--init", type=str) # calls installflowers17
parser.add_argument("-td", "--traindata", type=str) #Trainmodel
parser.add_argument("-tl", "--trainlabels", type=str) #Trainmodel
parser.add_argument("-p", "--predict", type=str) # predict(imagearray)

args = parser.parse_args()

#------------------------------------
# TUNEABLE PARAMETERS               |
#------------------------------------
images_per_class = 80
fixed_size       = tuple((500, 500))
train_path       = "golem/work/dataset/train"
h5_data          = 'h5/data.h5'
h5_labels        = 'h5/labels.h5'
bins             = 8
num_trees = 100
test_size = 0.10
seed      = 9
test_path  = "golem/work/dataset/test"
scoring    = "accuracy"
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
def trainmodel():
    # variables to hold the results and names
    results = []
    names   = []

    # import the feature vector and trained labels
    h5f_data  = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string   = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels   = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    # verify the shape of the feature vector and labels
    print("[STATUS] features shape: {}".format(global_features.shape))
    print("[STATUS] labels shape: {}".format(global_labels.shape))

    print("[STATUS] training started...")

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                            np.array(global_labels),
                                                                                            test_size=test_size,
                                                                                            random_state=seed)
    # create the model - Random Forests
    clf = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1,
                          #shuffle=True, n_iter=10,
                          verbose=1)
    # fit the training data to the model
    clf.partial_fit(trainDataGlobal, trainLabelsGlobal, classes=np.unique(trainLabelsGlobal))
    print("Model Fit")
    pkl_file = "classifier.pkl"
    with open(pkl_file, 'wb') as file:
        pickle.dump(clf, file)
    return True
def continuedTrain(h5data, h5labels):
        # variables to hold the results and names
    results = []
    names   = []

    # import the feature vector and trained labels
    h5f_data  = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string   = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels   = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                            np.array(global_labels),
                                                                                            test_size=test_size,
                                                                                            random_state=seed)
    # create the model - Random Forests
    with open("classifier.pkl", 'rb') as model:
        clf = pickle.load(model)
    # fit the training data to the model
    clf.partial_fit(trainDataGlobal, trainLabelsGlobal)
    print("Model Fit")
    pkl_file = "classifier.pkl"
    with open(pkl_file, 'wb') as file:
        pickle.dump(clf, file)
    print("Model Trained Additionally!")
def predict(target):
    train_labels = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
             "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
             "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
             "windflower", "pansy"]

    # sort the training labels
    train_labels.sort()
    # loop through the test images
    with open("classifier.pkl", 'rb') as model:
        clf = pickle.load(model)
    global_features = []
    # read the image
    image = cv2.imread(target)
    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image) 
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    global_features.append(global_feature)
    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_features)

    # predict label of test image
    prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    print(train_labels[prediction])
    return train_labels[prediction]
try:
    if len(args.traindata) > 3:
        continuedTrain(args.traindata, args.trainlabels)
except:
    pass
try:
    predict(args.predict)
except:
    pass

#py partialTrain.py --traindata "h5/data.h5" --trainlabels "h5/labels.h5"
#py partialTrain.py --predict "test2.jpg"
