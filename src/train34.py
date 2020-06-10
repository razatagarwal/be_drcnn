#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:05:33 2019

@author: razat_ag
"""

import tensorflow as tf
from keras.applications import InceptionV3
from keras.applications import ResNet50
import glob
import cv2
import numpy as np
from keras import optimizers
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import neptune
from keras.callbacks import Callback

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init_g)
    sess.run(init_l)

TR_0_folder = "training/3/"
TR_1_folder = "training/4/"

wd, ht = 100, 100
def datasetCreate(TR_0_folder, TR_1_folder):
    
    X_train =[]
    Y_train =[]
    
    for image_file in glob.iglob(TR_0_folder+  "*.png"):
        im = cv2.resize(cv2.imread(image_file),(wd, ht))
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        X_train.append(im)
        Y_train.append([1, 0])
        print(image_file)
    
    for image_file in glob.iglob(TR_1_folder+  "*.png"):
        im = cv2.resize(cv2.imread(image_file),(wd, ht))
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        X_train.append(im)
        Y_train.append([0, 1])
        print(image_file)
        
    return X_train, Y_train

X_train, labels = datasetCreate(TR_0_folder, TR_1_folder)
X_train = np.array(X_train)
labels = np.array(labels)
tr_indices = np.argmax(labels, axis = 1)

cw = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(np.argmax(labels, axis = 1)),
    y = np.argmax(labels, axis = 1))
print(cw)

a = optimizers.Adam(lr = 0.0005)

base_model = ResNet50(weights = None, 
    include_top = True, 
    input_shape = (wd, ht, 3),
    classes = 2)

for layer in base_model.layers:
    layer.trainable = True
	
base_model.compile(optimizer = a,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

neptune.init('razatagarwal/ML-PWT')

PARAMS = {
        'Name' : 'DR134',
        'Architecture' : 'ResNet101',
        'Learning_Rate' : 0.005,
        'Epochs' : 50}

class NeptuneMonitor(Callback):
    def on_epoch_end(self, epoch, logs={}):
        tr_accuracy = logs['acc']
        neptune.send_metric('tr_accuracy', epoch, tr_accuracy)
        val_accuracy = logs['val_acc']
        neptune.send_metric('val_accuracy', epoch, val_accuracy)

with neptune.create_experiment(name='DR134_RESNET50', params= PARAMS):
    neptune_monitor = NeptuneMonitor()
    history = base_model.fit(X_train,
        labels,
        batch_size = 32,
        epochs = 50,
        verbose = 1,
        shuffle = True,
        validation_split = 0.2,
        class_weight = cw,
        sample_weight = None,
        initial_epoch = 0)

new_modelJson = base_model.to_json()
with open("ResNet50_SEG34_E50.json", "w") as json_file:
    json_file.write(new_modelJson)
base_model.save_weights("ResNet50_SEG34_E50.h5")
print("Saved model to disk")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

result = base_model.predict(X_train, verbose = 1)
indices = np.argmax(result, axis = 1)

cm = confusion_matrix(tr_indices, indices)
print(cm)
acc_tr = accuracy_score(tr_indices, indices)
print(acc_tr)
