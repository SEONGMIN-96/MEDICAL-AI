from tensorflow.keras.applications import *
import os
from tensorflow import keras
import tensorflow as tf


#----------------------------------------------------------------------------

class LoadDenseNet():
    def __init__(self) -> None:
        ...    
    
    def densenet(self, inputs, backbone='DenseNet121', weights=None):
        
        # get last conv layer from the end of each block [28x28, 14x14, 7x7]
        if backbone == 'DenseNet121':
            inputs = tf.keras.applications.densenet.preprocess_input(inputs)
            model = DenseNet121(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'DenseNet169':
            inputs = tf.keras.applications.densenet.preprocess_input(inputs)
            model = DenseNet169(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'DenseNet201':
            inputs = tf.keras.applications.densenet.preprocess_input(inputs)
            model = DenseNet201(input_tensor=inputs, include_top=False, weights=weights)
              
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))
        
        # create the densenet backbone
        return model
    
    

