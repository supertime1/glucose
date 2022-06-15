""" This document complies all functions to do the tensorflow training """

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import TFModels
from TFConsts import TFConsts
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path


class TFTrainer:
    def __init__(self, train_data: np.array, train_label: np.array, output_dir: Path) -> None:
        self.train_data = train_data
        self.train_label = train_label
        self.input_shape = train_data.shape[1:]
        self.logs = {}
        self.training_begin_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard directory saves all training history from all models under the same model name
        self.tensorboard_log_dir = "tensorboard/fit/" + self.training_begin_time
        self.tf_const = TFConsts()
        
        if self.tf_const.model_for_training.RNN_ENCODER.value:
            self.model = TFModels.RNNEncoder(self.input_shape)
        if self.tf_const.model_for_training.RNN_ENCODER_DECODER.value:
            self.model = TFModels.RNNEncoderDecoder(self.input_shape, self.input_shape[0])
        
        self.output_dir = os.path.join(output_dir, 'logs', self.model.name) 
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
        os.makedirs(os.path.join(self.output_dir, self.training_begin_time))

        
    def train(self):
        tf.keras.backend.clear_session()
        self.model.compile(optimizer=self.tf_const.training_hyperparameters.OPTIMIZER.value,
                           loss=self.tf_const.training_hyperparameters.LOSS.value,
                           metrics=self.tf_const.training_hyperparameters.METRICS.value)
        self.model.build(input_shape=self.train_data.shape)        
        self.model.summary()

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=self.tf_const.training_hyperparameters.PATIENCE.value, verbose=0,
            mode='auto', baseline=None, restore_best_weights=True)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.output_dir, self.tensorboard_log_dir), update_freq='batch', histogram_freq=1)
        self.history = self.model.fit(self.train_data, self.train_label,
                                batch_size=self.tf_const.training_hyperparameters.BATCH_SIZE.value,
                                steps_per_epoch=len(self.train_data) // self.tf_const.training_hyperparameters.BATCH_SIZE.value,
                                epochs=self.tf_const.training_hyperparameters.EPOCH.value,
                                verbose=1,
                                validation_split=0.2,
                                callbacks=[early_stop, tensorboard_callback]
                                )
        
    def export_logs(self):
        # log data preprocessing steps
        self.logs['normalize_data_by_first_point'] = self.tf_const.data_preprocess_const.DATA_NORM_BY_FIRST_POINT.value
        self.logs['normalize_data_by_zscore'] = self.tf_const.data_preprocess_const.DATA_NORM_BY_ZSCORE.value
        self.logs['normalize_label_by_mean'] = self.tf_const.data_preprocess_const.LABEL_NORM_BY_MEAN.value
        
        # log model name and input and output shape
        self.logs['model'] = self.model.name
        self.logs['num_samples'] = self.train_data.shape[0]
        self.logs['input_shape'] = self.input_shape
        self.logs['output_shape'] = self.model.layers[-1].output_shape
        
        # log training hyperparameters
        self.logs['learning_rate'] = self.tf_const.training_hyperparameters.LEARNING_RATE.value
        self.logs['loss_fucntion'] = self.tf_const.training_hyperparameters.LOSS.value.name
        self.logs['metrics'] = self.tf_const.training_hyperparameters.METRICS.value
        self.logs['optimizer'] = self.tf_const.training_hyperparameters.OPTIMIZER.value._name
        self.logs['patience'] = self.tf_const.training_hyperparameters.PATIENCE.value
        self.logs['epoch'] = self.tf_const.training_hyperparameters.EPOCH.value
        self.logs['batch_size'] = self.tf_const.training_hyperparameters.BATCH_SIZE.value
        
        # log training performance
        self.logs['loss'] = self.history.history['loss']
        self.logs['val_loss'] = self.history.history['val_loss']
        
        # save json file
        jsonString = json.dumps(self.logs)
        jsonFile = open(os.path.join(self.output_dir, self.training_begin_time, "training_logs.json"), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def save_model(self):
        self.model.encoder.save(os.path.join(self.output_dir, self.training_begin_time, self.model.name + '.h5'))
