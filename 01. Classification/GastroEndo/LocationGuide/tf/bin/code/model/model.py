from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import layers

from .efficient import LoadEfficientNet
from .resnet import LoadResNet
from .vgg import LoadVGG
from .densenet import LoadDenseNet
from .inception import LoadInception
from .residual_attention import ResidualAttentionNetwork 
from .mobilenet import LoadMobileNet

import os
import tensorflow as tf
from tensorflow import keras


#----------------------------------------------------------------------------

class CreateGastricModel(LoadEfficientNet, LoadResNet, LoadVGG, LoadDenseNet, LoadInception, ResidualAttentionNetwork, LoadMobileNet):
    def __init__(self) -> None:
        ...    
            
    def load_imgShape(self, model_name):
        dict_modelIMGShape = {'EfficientNetB0':224,'EfficientNetB1':240,'EfficientNetB2':260,'EfficientNetB3':300,
                              'EfficientNetB4':380,'EfficientNetB5':456,'EfficientNetB6':528,'EfficientNetB7':600,
                              'resnet50':224,'resnet50V2':224,'resnet101':224,
                              'resnet101V2':224,'resnet152':224,'resnet152V2':224, 'vgg16':224, 'vgg19':224,
                              'InceptionResNetV2':299, 'InceptionV3':299, 'Residual_Attention':224, 'Residual92':224,
                              'MobileNet':224, 'MobileNetv2':224}
        
        return dict_modelIMGShape[model_name]
    
    
    def build_model(self, model_name, input_shape, n_class):
        
        # reference : https://keras.io/api/applications/
        allowed_effcinet = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                             'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']
        allowed_resnet = ['resnet50','resnet50V2','resnet101','resnet101V2','resnet152','resnet152V2']
        allowed_vgg = ['vgg16', 'vgg19']
        allowed_inception = ['InceptionV3', 'InceptionResNetV2', 'Xception']
        allowed_densenet = ['DenseNet121', 'DenseNet169', 'DenseNet201']
        allowed_residual_attention = ['Residual_Attention92', 'Residual92']
        allowed_mobilenet = ['MobileNet', 'MobileNetv2']
        
        if input_shape is None:
            if keras.backend.image_data_format() == 'channels_first':
                inputs = keras.layers.Input(shape=(3, None, None))
            else:
                inputs = keras.layers.Input(shape=(None, None, 3))
        else:
            inputs = keras.layers.Input(shape=input_shape)

        if model_name in allowed_resnet:
            base_model = self.resnet(inputs=inputs, backbone=model_name, weights='imagenet')

        elif model_name in allowed_effcinet:
            base_model = self.efficientnet(inputs=inputs, backbone=model_name, weights='imagenet')
            
        elif model_name in allowed_vgg:
            base_model = self.vgg(inputs=inputs, backbone=model_name, weights='imagenet')
            
        elif model_name in allowed_inception:
            base_model = self.inception(inputs=inputs, backbone=model_name, weights='imagenet')
            
        elif model_name in allowed_densenet:
            base_model = self.densenet(inputs=inputs, backbone=model_name, weights='imagenet')
        
        elif model_name in allowed_mobilenet:
            base_model = self.mobilenet(inputs=inputs, backbone=model_name, weights='imagenet')
        
        elif model_name in allowed_residual_attention:
            base_model = self.resattention(input_shape=input_shape, n_classes=n_class, backbone=model_name)

        # model = Sequential()
        # model.add(input_model)
        # model.add(layers.Flatten())
        # model.add(layers.Dense(1024, activation='relu'))
        # model.add(layers.Dense(1024, activation='relu'))
        # model.add(layers.Dense(n_class, activation='softmax'))
        # input_model.trainable = False
        
        # Freeze the pretrained weights
        for layer in base_model.layers:
            layer.trainable = False
            
        inputs = base_model.input   
        # x = img_augmentation(inputs)
        x = self.classifier_structure_GAP(base_model.output)
        
        if n_class==2:
            output = layers.Dense(n_class, activation='sigmoid', name='output')(x)
        elif n_class>2:
            output = layers.Dense(n_class, activation='softmax', name='output')(x)
            
        model = Model(inputs, output)
        print(model.summary())

        return model
            
    def load_model(self, exp_path):
        return load_model(os.path.join('bin', 'exp', exp_path, 'best_model.hdf5'), compile=False)
    
    def classifier_structure_GAP(self, x):
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Dropout(0.2)(x)
        return x
        
    def classifier_structure_F256x2(self, x):
        x = layers.GlobalAveragePooling2D()(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1024, activation='relu')(x)
        
        return x

    def classifier_structure_512x2(self, x):
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        
        return x
    
    #  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001)
    #              ,loss=tf.keras.losses.binary_crossentropy
    #              ,metrics=[tf.keras.metrics.Precision(name='precision')\
    #                       ,tf.keras.metrics.Recall(name='recall')\
    #                       ,tf.keras.metrics.FalsePositives(name='false_positives')\
    #                       ,tf.keras.metrics.FalseNegatives(name='false_negatives')])