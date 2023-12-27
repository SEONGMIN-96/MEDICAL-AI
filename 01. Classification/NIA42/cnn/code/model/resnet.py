# from tensorflow.keras.applications import ResNet50, ResNet101, InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import *
import os
from tensorflow import keras
import tensorflow as tf


#----------------------------------------------------------------------------

class LoadResNet():
    def __init__(self) -> None:
        ...    
    
    def resnet(self, inputs, backbone='resnet50', weights=None):
        # get last conv layer from the end of each block [28x28, 14x14, 7x7]
        if backbone == 'resnet50':
            inputs = tf.keras.applications.resnet.preprocess_input(inputs)
            model = ResNet50(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'resnet50V2':
            inputs = tf.keras.applications.resnet_v2.preprocess_input(inputs)
            model = ResNet50V2(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'resnet101':
            inputs = tf.keras.applications.resnet.preprocess_input(inputs)
            model = ResNet101(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'resnet101V2':
            inputs = tf.keras.applications.resnet_v2.preprocess_input(inputs)
            model = ResNet101V2(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'resnet152':
            inputs = tf.keras.applications.resnet.preprocess_input(inputs)
            model = ResNet152(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'resnet152V2':
            inputs = tf.keras.applications.resnet_v2.preprocess_input(inputs)
            model = ResNet152V2(input_tensor=inputs, include_top=False, weights=weights)
        elif backbone == 'InceptionResNetV2':
            inputs = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs)
            model = InceptionResNetV2(input_tensor=inputs, include_top=False, weights=weights)   
            
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))
        
        # create the densenet backbone
        return model