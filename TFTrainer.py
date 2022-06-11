""" This document complies all functions to do the tensorflow training """

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import TFModels
import TFConsts
import numpy as np
from datetime import datetime
import os
import json


class TFTrainer:
    def __init__(self, train_data: np.array, train_label: np.array, batch_size=128) -> None:
        self.train_data = train_data
        self.train_label = train_label
        self.input_shape = train_data.shape[1:]
        self.logs = {}
        self.log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        if TFConsts.ModelForTraining.RNN_ENCODER:
            self.model = TFModels.RNNEncoder(self.input_shape)
        if TFConsts.ModelForTraining.RNN_ENCODER_DECODER:
            self.model = TFModels.RNNEncoderDecoder(self.input_shape)
             
    def train(self):
        tf.keras.backend.clear_session()
        self.model.compile(optimizer=TFConsts.TrainingHyperparameters.OPTIMIZER,
                           loss=TFConsts.TrainingHyperparameters.LOSS,
                           metrics=TFConsts.TrainingHyperparameters.METRICS)
        self.model.build(input_shape=self.train_data.shape)        
        self.model.summary()

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=TFConsts.TrainingHyperparameters.PATIENCE, verbose=0,
            mode='auto', baseline=None, restore_best_weights=True)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, update_freq='batch', histogram_freq=1)
        history = self.model.fit(self.train_data, self.train_label,
                                batch_size=TFConsts.TrainingHyperparameters.BATCH_SIZE,
                                steps_per_epoch=len(self.train_data) // self.batch_size,
                                epochs=TFConsts.TrainingHyperparameters.EPOCH,
                                verbose=1,
                                validation_split=0.2,
                                callbacks=[early_stop, tensorboard_callback]
                                )
        
    def export_logs(self):
        # log data preprocessing steps
        self.logs['normalize_data_by_first_point'] = TFConsts.DataPreProcess.DATA_NORM_BY_FIRST_POINT
        self.logs['normalize_data_by_zscore'] = TFConsts.DataPreProcess.DATA_NORM_BY_ZSCORE
        self.logs['normalize_label_by_mean'] = TFConsts.DataPreProcess.LABEL_NORM_BY_MEAN
        
        # log model name and input and output shape
        self.logs['model'] = self.model.name
        self.logs['num_samples'] = self.train_data.shape[0]
        self.logs['input_shape'] = self.input_shape
        self.logs['output_shape'] = self.model.output_shape
        
        # log training hyperparameters
        self.logs['learning_rate'] = TFConsts.TrainingHyperparameters.LEARNING_RATE
        self.logs['loss_fucntion'] = TFConsts.TrainingHyperparameters.LOSS.name
        self.logs['metrics'] = TFConsts.TrainingHyperparameters.METRICS
        self.logs['optimizer'] = TFConsts.TrainingHyperparameters.OPTIMIZER.name
        self.logs['patience'] = TFConsts.TrainingHyperparameters.PATIENCE
        self.logs['epoch'] = TFConsts.TrainingHyperparameters.EPOCH
        self.logs['batch_size'] = TFConsts.TrainingHyperparameters.BATCH_SIZE
        
        # TODO: continue here
        jsonString = json.dumps(self.logs)
        jsonFile = open("training_logs.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

