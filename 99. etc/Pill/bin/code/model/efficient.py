from tensorflow.keras.applications import *
import os
from tensorflow import keras
import tensorflow as tf


#----------------------------------------------------------------------------

class LoadEfficientNet():
    def __init__(self) -> None:
        ...    
    
    def efficientnet(self, inputs, backbone='EfficientNetB0', weights=None):
        inputs = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # get last conv layer from the end of each block [28x28, 14x14, 7x7]
        if backbone == 'EfficientNetB0':
            model = EfficientNetB0(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'EfficientNetB1':
            model = EfficientNetB1(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'EfficientNetB2':
            model = EfficientNetB2(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'EfficientNetB3':
            model = EfficientNetB3(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'EfficientNetB4':
            model = EfficientNetB4(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'EfficientNetB5':
            model = EfficientNetB5(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'EfficientNetB6':
            model = EfficientNetB6(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'EfficientNetB7':
            model = EfficientNetB7(input_tensor=inputs, include_top=False, weights=weights)
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))
        
        # create the densenet backbone
        return model