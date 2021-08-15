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
import keras.layers.convolutional 
from sklearn.metrics import confusion_matrix
import rpyc.core.service 
import itertools
import os
import argparse
import rpyc
from rpyc.utils.server import ThreadedServer
from threading import Thread

class ClassifierService(rpyc.Service):
    def exposed_buildModel(self, train_path, valid_path, classes):
        train_batches = ImageDataGenerator().flow_from_directory(
            train_path, target_size=(224, 224), classes=classes, batch_size=10)
        valid_batches = ImageDataGenerator().flow_from_directory(
            valid_path, target_size=(224, 224), classes=classes, batch_size=4)

        vgg16_model = keras.applications.vgg16.VGG16(
            weights="/golem/work/vgg16.h5")

        model = Sequential()
        for i in vgg16_model.layers:
            model.add(i)
        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(4, activation='softmax'))
        model.compile(
            Adam(
                lr=0.07),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        model.fit_generator(
            train_batches,
            steps_per_epoch=1,
            validation_data=valid_batches,
            validation_steps=1,
            epochs=30,
            verbose=2)
        server.fmodel = model
        return True

    def exposed_predict(self, testloc):
        model = server.fmodel
        test_batches = ImageDataGenerator().flow_from_directory(testloc, target_size=(
            224, 224), classes=["Unknown"], batch_size=1, save_format='jpg')
        test_imgs, test_labels = next(test_batches)
        predictions = model.predict_on_batch(np.array(test_imgs))
        test_labels = np.array(test_labels.argmax(axis=1))
        predictions = np.array(predictions.argmax(axis=1))
        # print(test_labels) Ignored because its wrong
        return predictions

    def exposed_train(self, classes, trainloc, validloc):
        model = server.fmodel
        train_batches = ImageDataGenerator().flow_from_directory(
            trainloc, target_size=(224, 224), classes=classes, batch_size=10)
        valid_batches = ImageDataGenerator().flow_from_directory(
            validloc, target_size=(224, 224), classes=classes, batch_size=4)
        model.fit_generator(
            train_batches,
            steps_per_epoch=1,
            validation_data=valid_batches,
            validation_steps=1,
            epochs=30,
            verbose=2)
        server.fmodel = model
        return True

if __name__ == "__main__":
    # Now time to start the server

    server = ThreadedServer(
        ClassifierService,
        socket_path='/golem/run/uds_socket')
    # Attach model to server
    t = Thread(target=server.start)
    t.daemon = True
    t.start()
    t.join()
