from tensorflow.keras.applications import *
import os
from tensorflow import keras
import tensorflow as tf


#----------------------------------------------------------------------------

class LoadInception():
    def __init__(self) -> None:
        ...    
    
    def inception(self, inputs, backbone='InceptionV3', weights=None):
        if backbone == 'InceptionV3':
            inputs = tf.keras.applications.inception_v3.preprocess_input(inputs)
            model = InceptionV3(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'InceptionResNetV2':
            inputs = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs)
            model = InceptionResNetV2(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'Xception':
            inputs = tf.keras.applications.xception.preprocess_input(inputs)
            model = Xception(input_tensor=inputs, include_top=False, weights=weights)
            
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))
        
        # create the densenet backbone
        return model