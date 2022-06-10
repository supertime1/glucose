"""  This document setup tensorflow training parameters and hyperparameters """

from enum import Enum


class DataPreProcess(Enum):
    DATA_NORM_BY_FIRST_POINT = True
    DATA_NORM_BY_ZSCORE = False
    LABEL_NORM_BY_MEAN = True


class ModelForTraining(Enum):
    TSMLENCODER = True

class TrainingHyperparameters(Enum):
    LEARNING_RATE = 0.001
    

class TFConsts:
    def __init__(self) -> None:
        self.data_preprocess_const = DataPreProcess
        self.model_for_training = ModelForTraining

    