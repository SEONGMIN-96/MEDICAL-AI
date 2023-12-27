from tensorflow.keras.applications import *
import os
from tensorflow import keras
import tensorflow as tf


#----------------------------------------------------------------------------

class LoadMobileNet():
    def __init__(self) -> None:
        ...    
    
    def mobilenet(self, inputs, backbone='MobileNet', weights=None):
        
        # get last conv layer from the end of each block [28x28, 14x14, 7x7]
        if backbone == 'MobileNet':
            inputs = tf.keras.applications.mobilenet.preprocess_input(input)
            model = MobileNet(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'MobileNetv2':
            inputs = tf.keras.applications.mobilenet_v2.preprocess_input(input)
            model = MobileNetV2(input_tensor=inputs, include_top=False, weights=weights)
            
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))
        
        # create the densenet backbone
        return model