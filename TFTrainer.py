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
    def __init__(self, train_data: np.array, train_label: np.array, output_dir: Path, note='') -> None:
        self.train_data = train_data
        self.train_label = train_label
        self.input_shape = train_data.shape[1:]
        self.note = note # add fold index here
        self.logs = {}
        self.training_begin_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard directory saves all training history from all models under the same model name
        self.tensorboard_log_dir = os.path.join("logs", self.training_begin_time)
        self.tf_const = TFConsts()
        
        if self.tf_const.model_for_training.RNN_ENCODER:
            self.model = TFModels.RNNEncoder(self.input_shape)
        if self.tf_const.model_for_training.RNN_ENCODER_DECODER:
            self.model = TFModels.RNNEncoderDecoder(self.input_shape)
        if self.tf_const.model_for_training.AUTO_ENCODER_TSML:
            self.model = TFModels.AutoEncoderTSML(self.input_shape[0], self.input_shape[1], 4)
            self.train_label = self.train_data

        
        self.output_dir = os.path.join(output_dir, 'logs', self.model.name) 
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
        os.makedirs(os.path.join(self.output_dir, self.training_begin_time))

        
    def train(self):
        tf.keras.backend.clear_session()
        # reset the model weights to prevent retraining the same model by using different folds of data
        weight_loaded = False
        
        # load empty weights to guanrantee a fresh training during cross validation loop
        if weight_loaded:
            self.model.load_weights('initial_weights.h5')
            weight_loaded = True
            print('reinitialized weights')
            
        self.model.compile(optimizer=self.tf_const.training_hyperparameters.OPTIMIZER,
                           loss=self.tf_const.training_hyperparameters.LOSS,
                           metrics=self.tf_const.training_hyperparameters.METRICS)

            
        self.model.build(input_shape=self.train_data.shape)        
        self.model.summary()
        
        if not weight_loaded:
            self.model.save_weights('initial_weights.h5')
            print('Saving initial weights')

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=self.tf_const.training_hyperparameters.PATIENCE, verbose=0,
            mode='auto', baseline=None, restore_best_weights=True)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.output_dir, self.tensorboard_log_dir), update_freq='batch', histogram_freq=1)
        
        self.history = self.model.fit(self.train_data, self.train_label,
                                        batch_size=self.tf_const.training_hyperparameters.BATCH_SIZE,
                                        steps_per_epoch=len(self.train_data) // self.tf_const.training_hyperparameters.BATCH_SIZE,
                                        epochs=self.tf_const.training_hyperparameters.EPOCH,
                                        verbose=1,
                                        validation_split=0.2,
                                        callbacks=[early_stop, tensorboard_callback]
                                        )
        
    def export_logs(self):
        # log data preprocessing steps
        self.logs['normalize_data_by_first_point'] = self.tf_const.data_preprocess_const.DATA_NORM_BY_FIRST_POINT
        self.logs['normalize_data_by_zscore'] = self.tf_const.data_preprocess_const.DATA_NORM_BY_ZSCORE
        self.logs['normalize_label_by_mean'] = self.tf_const.data_preprocess_const.LABEL_NORM_BY_MEAN
        
        # log model name and input and output shape
        self.logs['model'] = self.model.name
        self.logs['num_samples'] = self.train_data.shape[0]
        self.logs['input_shape'] = self.input_shape
        self.logs['output_shape'] = self.model.layers[-1].output_shape
        self.logs['note'] = self.note
        
        # log training hyperparameters
        self.logs['learning_rate'] = self.tf_const.training_hyperparameters.LEARNING_RATE
        self.logs['loss_fucntion'] = self.tf_const.training_hyperparameters.LOSS.name
        self.logs['metrics'] = self.tf_const.training_hyperparameters.METRICS
        self.logs['optimizer'] = self.tf_const.training_hyperparameters.OPTIMIZER._name
        self.logs['patience'] = self.tf_const.training_hyperparameters.PATIENCE
        self.logs['epoch'] = self.tf_const.training_hyperparameters.EPOCH
        self.logs['batch_size'] = self.tf_const.training_hyperparameters.BATCH_SIZE
        
        # log training performance
        self.logs['loss'] = self.history.history['loss']
        self.logs['val_loss'] = self.history.history['val_loss']
        
        # save json file
        jsonString = json.dumps(self.logs)
        jsonFile = open(os.path.join(self.output_dir, self.training_begin_time, f"training_logs_{self.note}.json"), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def save_model(self):
        try:
            self.model.encoder.save(os.path.join(self.output_dir, self.training_begin_time, self.model.name + f'_{self.note}.h5'))
        except Exception as exc:
            print(f'exception {exc}')
            self.model.encoder.encoder.save(os.path.join(self.output_dir, self.training_begin_time, self.model.name + f'_{self.note}.h5'))
