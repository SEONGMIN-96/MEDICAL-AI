from tensorflow.keras.applications import *
import os
from tensorflow import keras
import tensorflow as tf


#----------------------------------------------------------------------------

class LoadVGG():
    def __init__(self) -> None:
        ...    
    
    def vgg(self, inputs, backbone='vgg19', weights=None):
        
        # get last conv layer from the end of each block [28x28, 14x14, 7x7]
        if backbone == 'vgg16':
            inputs = tf.keras.applications.vgg16.preprocess_input(inputs)
            model = VGG16(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'vgg19':
            inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
            model = VGG19(input_tensor=inputs, include_top=False, weights=weights)
            
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))
        
        # create the densenet backbone
        return model