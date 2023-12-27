from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, AveragePooling2D, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from .blocks import residual_block
from .blocks import attention_block

class ResidualAttentionNetwork():
    def __init__(self) -> None:
        ...    
    
    def resattention(self, input_shape, n_classes, backbone):
        if backbone == 'Residual_Attention92':
            model = self.AttentionResNet92(n_classes=n_classes, shape=input_shape)
        elif backbone == 'Residual92':
            model = self.ResNet92(n_classes=n_classes,  shape=input_shape)
        
        return model
    
    def AttentionResNet92(self, n_classes, shape, n_channels=64, dropout=0.2, regularization=0.01):
        """
        Attention-92 ResNet
        https://arxiv.org/abs/1704.06904
        """
        regularizer = l2(regularization)

        input_ = Input(shape=shape)
        x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

        x = residual_block(x, output_channels=n_channels * 4)  # 56x56
        x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

        x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
        x = attention_block(x, encoder_depth=2)  # bottleneck 7x7
        x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

        x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
        x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
        x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
        x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

        x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
        x = residual_block(x, output_channels=n_channels * 32)
        x = residual_block(x, output_channels=n_channels * 32)

        pool_size = (x.get_shape()[1], x.get_shape()[2])
        # pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
        x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
        x = Flatten()(x)
        
        # if dropout:
            # x = Dropout(dropout)(x)
        
        output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

        model = Model(input_, output)
        
        return model
    
    def ResNet92(self, n_classes, shape=(224, 224, 3), n_channels=64, dropout=0.2, regularization=0.01):
        """
        ResNet
        https://arxiv.org/abs/1704.06904
        """
        regularizer = l2(regularization)

        input_ = Input(shape=shape)
        x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

        x = residual_block(x, output_channels=n_channels * 4)  # 56x56

        x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28

        x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14

        x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
        x = residual_block(x, output_channels=n_channels * 32)
        x = residual_block(x, output_channels=n_channels * 32)

        pool_size = (x.get_shape()[1], x.get_shape()[2])
        # pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
        x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
        x = Flatten()(x)
        
        # if dropout:
            # x = Dropout(dropout)(x)
        
        output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

        model = Model(input_, output)
        
        return model


    def AttentionResNet56(self, shape=(224, 224, 3), n_channels=64, n_classes=100,
                        dropout=0, regularization=0.01):
        """
        Attention-56 ResNet
        https://arxiv.org/abs/1704.06904
        """

        regularizer = l2(regularization)

        input_ = Input(shape=shape)
        x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

        x = residual_block(x, output_channels=n_channels * 4)  # 56x56
        x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

        x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
        x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

        x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
        x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

        x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
        x = residual_block(x, output_channels=n_channels * 32)
        x = residual_block(x, output_channels=n_channels * 32)

        pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
        x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
        x = Flatten()(x)
        if dropout:
            x = Dropout(dropout)(x)
        output = Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

        model = Model(input_, output)
        return model


    def AttentionResNetCifar10(self, shape=(32, 32, 3), n_channels=32, n_classes=10):
        """
        Attention-56 ResNet for Cifar10 Dataset
        https://arxiv.org/abs/1704.06904
        """
        input_ = Input(shape=shape)
        x = Conv2D(n_channels, (5, 5), padding='same')(input_)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)  # 16x16

        x = residual_block(x, input_channels=32, output_channels=128)
        x = attention_block(x, encoder_depth=2)

        x = residual_block(x, input_channels=128, output_channels=256, stride=2)  # 8x8
        x = attention_block(x, encoder_depth=1)

        x = residual_block(x, input_channels=256, output_channels=512, stride=2)  # 4x4
        x = attention_block(x, encoder_depth=1)

        x = residual_block(x, input_channels=512, output_channels=1024)
        x = residual_block(x, input_channels=1024, output_channels=1024)
        x = residual_block(x, input_channels=1024, output_channels=1024)

        x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(x)  # 1x1
        x = Flatten()(x)
        output = Dense(n_classes, activation='softmax')(x)

        model = Model(input_, output)
        return model