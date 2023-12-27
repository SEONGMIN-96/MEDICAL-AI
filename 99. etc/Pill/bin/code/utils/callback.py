from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler

import tensorflow as tf

import os
import numpy as np
import glob
import cv2


#----------------------------------------------------------------------------

class CallBack():
    def __init__(self) -> None:
        ...
        
    def callback_setting(self, es_patience: int, now_time: str, reduce_lr_patience: int, reduce_lr_factor: float):
        """
            
        Args:
            ...

        Return:
            ...
        """
        es = EarlyStopping(monitor='val_loss', mode='auto', patience=es_patience)
        cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                            filepath=os.path.join('bin', 'exp', now_time, 'best_model.hdf5'))
        
        # learning_rate_scheduler
        if self.learning_rate_scheduler == 'ReduceLROnPlateau':
            lrs = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr_patience, mode='auto', verbose=1, factor=reduce_lr_factor)
        elif self.learning_rate_scheduler == 'CosineDecay':
            cos_decay = tf.keras.experimental.CosineDecay(initial_learning_rate=0.001, decay_steps=50, alpha=0.001)
            lrs = LearningRateScheduler(cos_decay, verbose=1)
        
        return [es, lrs, cp]
    
    def aaa(self):
        ...