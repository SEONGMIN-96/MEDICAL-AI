from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler

from datetime import datetime

import tensorflow as tf

import os
import numpy as np
import glob
import cv2


#----------------------------------------------------------------------------

class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = tf.timestamp()
    def on_epoch_end(self, epoch, logs = {}):
        time = tf.timestamp() - self.timetaken
        self.times.append(time)
        self.epochs.append(epoch)
        
        print('\n{}, Epoch {} ended.'.format(time, epoch + 1))
    def on_train_end(self, logs = {}):
        ...