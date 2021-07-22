#!/usr/local/bin/python
import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.applications import vgg16
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import argparse
import rpyc
from rpyc.utils.server import ThreadedServer
from threading import Thread
train_path = 'cats-dogs-monkeys-cows/train'
valid_path = 'cats-dogs-monkeys-cows/valid'
test_path = 'cats-dogs-monkeys-cows/test'
def buildModel(train_path, valid_path, classes):
        train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=classes, batch_size=10)
        valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=classes, batch_size=4)

        imgs, labels = next(train_batches)

        train_batches.class_indices

        vgg16_model = keras.applications.vgg16.VGG16()

        model = Sequential()
        for i in vgg16_model.layers:
            model.add(i)
        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(4, activation='softmax'))
        model.compile(Adam(lr=0.07), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(train_batches, steps_per_epoch=1, 
                            validation_data=valid_batches, validation_steps=1, epochs=30, verbose=2)
        return model
class ClassifierService(rpyc.Service):
    def exposed_predict(self, classes, testloc):
        model = server.fmodel
        test_batches = ImageDataGenerator().flow_from_directory(testloc, target_size=(224,224), classes=["Unknown"], batch_size=1,save_format='jpg')
        test_imgs, test_labels = next(test_batches)
        predictions = model.predict_on_batch(np.array(test_imgs))
        test_labels = np.array(test_labels.argmax(axis=1))
        predictions = np.array(predictions.argmax(axis=1))
        #print(test_labels) Ignored because its wrong
        return predictions
        
#model = buildModel(train_path, valid_path, ['dog','cat','monkey','cow'])


if __name__ == "__main__":
    #Main Runtime loop
    parser = argparse.ArgumentParser()
    #Required Args to train a model
    parser.add_argument("-v", "--validationpath", type=str) 
    parser.add_argument("-t", "--trainpath", type=str) 
    parser.add_argument("-c", "--classes", nargs='+')
    #Batch can be changed in theory, but it really only makes sense on very powerful machines
    #So I figure those who need it will see this and go change it

    args = parser.parse_args()
    if args.validationpath == None:
        print("Please Provide a Validation Path, , via -v <path>")
    if args.trainpath == None:
        print("Please Provide a Training Path, via -t <path>")
    if args.validationpath == None:
        print("Please Provide Classes Path, via -c <array>")
    print("{args.classes}")
    fModel = buildModel(args.trainpath, args.validationpath, args.classes)
    # Now time to start the server
    server = ThreadedServer(ClassifierService, port = 12345)
    # Attach model to server
    server.fmodel = fModel
    t = Thread(target = server.start)
    t.daemon = True
    t.start()
    t.join()
    #With model built, 

