from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import layers

from .efficient import LoadEfficientNet
from .resnet import LoadResNet
from .vgg import LoadVGG
from .densenet import LoadDenseNet
from .inception import LoadInception

import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import concatenate, Concatenate


#----------------------------------------------------------------------------

class CreateOTEModel_v2(LoadEfficientNet, LoadResNet, LoadVGG, LoadDenseNet, LoadInception):
    def __init__(self) -> None:
        ...    
            
    def load_imgShape(self, model_name):
        dict_modelIMGShape = {'EfficientNetB0':224,'EfficientNetB1':240,'EfficientNetB2':260,'EfficientNetB3':300,
                              'EfficientNetB4':380,'EfficientNetB5':456,'EfficientNetB6':528,'EfficientNetB7':600,
                              'resnet50':224,'resnet50V2':224,'resnet101':256, #224,
                              'resnet101V2':224,'resnet152':224,'resnet152V2':224, 'vgg16':224, 'vgg19':224,
                              'InceptionResNetV2':256, 'InceptionV3':256}
        
        return dict_modelIMGShape[model_name]
    
    
    def build_model(self, model_name, input_shape, n_class):
        
        # reference : https://keras.io/api/applications/
        allowed_effcinet = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                             'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']
        allowed_resnet = ['resnet50','resnet50V2','resnet101','resnet101V2','resnet152','resnet152V2']
        allowed_vgg = ['vgg16', 'vgg19']
        allowed_inception = ['InceptionV3', 'InceptionResNetV2', 'Xception']
        allowed_densenet = ['DenseNet121', 'DenseNet169', 'DenseNet201']
        
        print(model_name, input_shape)

        if input_shape is None:
            if keras.backend.image_data_format() == 'channels_first':
                inputs0 = keras.layers.Input(shape=(3, None, None))
                inputs1 = keras.layers.Input(shape=(3, None, None))
                inputs2 = keras.layers.Input(shape=(3, None, None))
                inputs3 = keras.layers.Input(shape=(3, None, None))
                inputs4 = keras.layers.Input(shape=(3, None, None))
            else:
                inputs0 = keras.layers.Input(shape=(None, None, 3))
                inputs1 = keras.layers.Input(shape=(None, None, 3))
                inputs2 = keras.layers.Input(shape=(None, None, 3))
                inputs3 = keras.layers.Input(shape=(None, None, 3))
                inputs4 = keras.layers.Input(shape=(None, None, 3))
        else:
            inputs0 = keras.layers.Input(shape=input_shape)
            inputs1 = keras.layers.Input(shape=input_shape)
            inputs2 = keras.layers.Input(shape=input_shape)
            inputs3 = keras.layers.Input(shape=input_shape)
            inputs4 = keras.layers.Input(shape=input_shape)

        if model_name in allowed_resnet:
            base_model0 = self.resnet(inputs=inputs0, backbone=model_name, weights='imagenet')
            base_model1 = self.resnet(inputs=inputs1, backbone=model_name, weights='imagenet')
            base_model2 = self.resnet(inputs=inputs2, backbone=model_name, weights='imagenet')
            base_model3 = self.resnet(inputs=inputs3, backbone=model_name, weights='imagenet')
            base_model4 = self.resnet(inputs=inputs4, backbone=model_name, weights='imagenet')

        elif model_name in allowed_effcinet:
            base_model0 = self.efficientnet(inputs=inputs0, backbone=model_name, weights='imagenet')
            base_model1 = self.efficientnet(inputs=inputs1, backbone=model_name, weights='imagenet')
            base_model2 = self.efficientnet(inputs=inputs2, backbone=model_name, weights='imagenet')
            base_model3 = self.efficientnet(inputs=inputs3, backbone=model_name, weights='imagenet')
            base_model4 = self.efficientnet(inputs=inputs4, backbone=model_name, weights='imagenet')
            
        elif model_name in allowed_vgg:
            base_model0 = self.vgg(inputs=inputs0, backbone=model_name, weights='imagenet')
            base_model1 = self.vgg(inputs=inputs1, backbone=model_name, weights='imagenet')
            base_model2 = self.vgg(inputs=inputs2, backbone=model_name, weights='imagenet')
            base_model3 = self.vgg(inputs=inputs3, backbone=model_name, weights='imagenet')
            base_model4 = self.vgg(inputs=inputs4, backbone=model_name, weights='imagenet')
            
        elif model_name in allowed_inception:
            base_model0 = self.inception(inputs=inputs0, backbone=model_name, weights='imagenet')
            base_model1 = self.inception(inputs=inputs1, backbone=model_name, weights='imagenet')
            base_model2 = self.inception(inputs=inputs2, backbone=model_name, weights='imagenet')
            base_model3 = self.inception(inputs=inputs3, backbone=model_name, weights='imagenet')
            base_model4 = self.inception(inputs=inputs4, backbone=model_name, weights='imagenet')
            
        elif model_name in allowed_densenet:
            base_model0 = self.densenet(inputs=inputs0, backbone=model_name, weights='imagenet')
            base_model1 = self.densenet(inputs=inputs1, backbone=model_name, weights='imagenet')
            base_model2 = self.densenet(inputs=inputs2, backbone=model_name, weights='imagenet')
            base_model3 = self.densenet(inputs=inputs3, backbone=model_name, weights='imagenet')
            base_model4 = self.densenet(inputs=inputs4, backbone=model_name, weights='imagenet')

        # Freeze the pretrained weights
        for layer in base_model0.layers:
            layer.trainable = False
        
        for layer in base_model0.layers:
            layer._name = "model_0_" + layer.name
        for layer in base_model1.layers:
            layer._name = "model_1_" + layer.name
        for layer in base_model2.layers:
            layer._name = "model_2_" + layer.name
        for layer in base_model3.layers:
            layer._name = "model_3_" + layer.name
        for layer in base_model4.layers:
            layer._name = "model_4_" + layer.name
        
        base_inputs0 = base_model0.input   
        base_inputs1 = base_model1.input   
        base_inputs2 = base_model2.input   
        base_inputs3 = base_model3.input   
        base_inputs4 = base_model4.input
        
        # concatenate 레이어를 사용하여 N개의 입력 텐서를 병합
        merged = concatenate([base_model0.output, base_model1.output, base_model2.output, base_model3.output, base_model4.output], axis=-1)
        
        merged_output = self.classifier_structure_GAP(merged)
                
        if n_class==2:
            output = layers.Dense(n_class, activation='sigmoid', name='output')(merged_output)
        elif n_class>2:
            output = layers.Dense(n_class, activation='softmax', name='output')(merged_output)
        
        inputs = [base_inputs0, base_inputs1, base_inputs2, base_inputs3, base_inputs4]
        
        model = Model(inputs, output)
        # print(model.summary())

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