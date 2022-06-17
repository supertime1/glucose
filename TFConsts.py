"""  This document setup tensorflow training parameters and hyperparameters """

from enum import Enum
import tensorflow as tf


class DataPreProcess(Enum):
    DATA_NORM_BY_FIRST_POINT = True
    DATA_NORM_BY_ZSCORE = False
    LABEL_NORM_BY_MEAN = True


class ModelForTraining(Enum):
    RNN_ENCODER = False
    AUTO_ENCODER_TSML = True
    RNN_ENCODER_DECODER = False

class TrainingHyperparameters(Enum):
    LEARNING_RATE = 0.001
    LOSS = tf.keras.losses.MeanSquaredError()
    OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
    METRICS = ['mae']
    PATIENCE = 20
    EPOCH = 200
    BATCH_SIZE = 128

class TFConsts:
    def __init__(self) -> None:
        self.data_preprocess_const = DataPreProcess
        self.model_for_training = ModelForTraining
        self.training_hyperparameters = TrainingHyperparameters

    