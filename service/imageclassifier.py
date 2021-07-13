#!/usr/local/bin/python
import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
from keras.api._v1.keras.applications.vgg16 import *
import argparse
import warnings
warnings.filterwarnings('ignore')
#Define an arg parsers to build out a CLI
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--init", type=str) # calls installflowers17
parser.add_argument("-t", "--trainloc", type=str) #Trainmodel
parser.add_argument("-b", "--batch", type=str) # predict(imagearray)
parser.add_argument("-p", "--predict", type=str) # predict(imagearray)

args = parser.parse_args()

def predict(test_path, batchnumber):
    classesf = ["bluebell", "buttercup", "coltsfoot", "cowslip"]
    test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=["Unknown"], batch_size=batchnumber,save_format='jpg')
    model = keras.models.load_model("/golem/work/model")
    test_imgs, test_labels = next(test_batches)
    predictions = model.predict_on_batch(np.array(test_imgs))
    test_labels = np.array(test_labels.argmax(axis=1))
    predictions = np.array(predictions.argmax(axis=1))
    #print(test_labels) Ignored because its wrong
    print(predictions)
def continuedTrain(train_path):
    classesf = ["bluebell", "buttercup", "coltsfoot", "cowslip"]
    model = keras.models.load_model("/golem/work/model")
    valid_path = train_path 
    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=classesf, batch_size=5, save_format='jpg')
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=classesf, batch_size=5, save_format='jpg')
    model.fit_generator(train_batches, steps_per_epoch=1, 
                    validation_data=valid_batches, validation_steps=1, epochs=30, verbose=2)
    model.save('/golem/work/model')
#continuedTrain('dataset/train')

if args.trainloc != None:
    continuedTrain(args.trainloc)

if args.predict != None:
    predict(args.predict, int(args.batch))

