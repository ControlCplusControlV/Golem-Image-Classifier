#-----------------------------------------
# Initialize Main dataset
#-----------------------------------------
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
#Define an arg parsers to build out a CLI
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--init", type=str) # calls installflowers17
parser.add_argument("-t", "--trainmodel", type=str) #Trainmodel
parser.add_argument("-p", "--predict", type=str) # predict(imagearray)

args = parser.parse_args()

def download_dataset(filename, url, work_dir):
	if not os.path.exists(filename):
		print("[INFO] Downloading flowers17 dataset....")
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
		statinfo = os.stat(filename)
		print("[INFO] Succesfully downloaded " + filename + " " + str(statinfo.st_size) + " bytes.")
		untar(filename, work_dir)

def jpg_files(members):
	for tarinfo in members:
		if os.path.splitext(tarinfo.name)[1] == ".jpg":
			yield tarinfo

def untar(fname, path):
	tar = tarfile.open(fname)
	tar.extractall(path=path, members=jpg_files(tar))
	tar.close()
	print("[INFO] Dataset extracted successfully.")

#-------------------------
# MAIN FUNCTION
#-------------------------
def installflowers17():
	flowers17_url  = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/"
	flowers17_name = "17flowers.tgz"
	train_dir      = "/home/dataset"

	if not os.path.exists(train_dir):
		os.makedirs(train_dir + "/train")

	download_dataset(flowers17_name, flowers17_url, train_dir)
	if os.path.exists(train_dir + "/jpg"):
		os.rename(train_dir + "/jpg", train_dir + "/train")


	# get the class label limit
	class_limit = 17

	# take all the images from the dataset
	image_paths = (train_dir + "/train/*.jpg")

	# variables to keep track
	label = 0
	i = 0
	j = 80

	# flower17 class names
	class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
			   	   "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
			       "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
			       "windflower", "pansy"]

	# loop over the class labels
	for x in range(1, class_limit+1):
		# create a folder for that class
		os.makedirs(train_dir + "/train/" + class_names[label])
		# get the current path
		cur_path = train_dir + "/train/" + class_names[label] + "/"
		# loop over the images in the dataset
		for index, image_path in enumerate(image_paths[i:j], start=1):
			original_path   = image_path
			image_path      = image_path.split("/")
			image_file_name = str(index) + ".jpg"
			os.rename(original_path, cur_path + image_file_name)
		
		i += 80
		j += 80
		label += 1
#------------------------------------
# TUNEABLE PARAMETERS               |
#------------------------------------
images_per_class = 80
fixed_size       = tuple((500, 500))
train_path       = "/home/dataset/train"
h5_data          = '/home/output/data.h5'
h5_labels        = '/home/output/labels.h5'
bins             = 8
num_trees = 100
test_size = 0.10
seed      = 9
test_path  = "dataset/test"
scoring    = "accuracy"
# feature-descriptor-1: Hu Moments
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
    # get the training labels
    train_labels = os.listdir(train_path)

    # sort the training labels
    train_labels.sort()
    print(train_labels)

    # empty lists to hold feature vectors and labels
    global_features = []
    labels          = []

    # loop over the training data sub-folders
    for training_name in train_labels:
        # join the training data path and each species training folder
        dir = os.path.join(train_path, training_name)

        # get the current training label
        current_label = training_name

        # loop over the images in each sub-folder
        for x in range(1,images_per_class+1):
            # get the image file name
            file = dir + "/" + str(x) + ".jpg"

            # read the image and resize it to a fixed-size
            image = cv2.imread(file)
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

            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

        print("[STATUS] processed folder: {}".format(current_label))

    print("[STATUS] completed Global Feature Extraction...")

    # get the overall feature vector size
    print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

    # get the overall training label size
    print("[STATUS] training Labels {}".format(np.array(labels).shape))

    # encode the target labels
    targetNames = np.unique(labels)
    le          = LabelEncoder()
    target      = le.fit_transform(labels)
    print("[STATUS] training labels encoded...")

    # scale features in the range (0-1)
    scaler            = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print("[STATUS] feature vector normalized...")

    print("[STATUS] target labels: {}".format(target))
    print("[STATUS] target labels shape: {}".format(target.shape))

    # save the feature vector using HDF5
    h5f_data = h5py.File(h5_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File(h5_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    print("[STATUS] end of training..")


    # get the training labels
    train_labels = os.listdir(train_path)

    # sort the training labels
    train_labels.sort()

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # create all the machine learning models
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))

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

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(trainDataGlobal.shape))
    print("Test data   : {}".format(testDataGlobal.shape))
    print("Train labels: {}".format(trainLabelsGlobal.shape))
    print("Test labels : {}".format(testLabelsGlobal.shape))
    # 10-fold cross validation
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    #-----------------------------------
    # TESTING OUR MODEL
    #-----------------------------------

    # to visualize results
    import matplotlib.pyplot as plt

    # create the model - Random Forests
    clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)
    pkl_file = "classifier.pkl"
    with open(pkl_file, 'wb') as file:
        pickle.dump(clf, file)
    return True
def predict(target):
    train_labels = os.listdir(train_path)

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

    # display the output image
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.show()
    return train_labels[prediction]
#predict([r"B:\\Golem Image Classifier\\dataset\\test\\44.jpg"])
#"/home/dataset/test/44.jpg"
if args.trainmodel == "True":
    trainmodel()
elif args.init == "True":
    installflowers17()
elif args.predict == "True":
    predict()
