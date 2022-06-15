""" 
This document includes functions to prepare training for tensorflow in cloud
"""

import numpy as np
from pathlib import Path
import os
import pickle
from typing import Tuple
import TFConsts


class TFDataPreprocessor:
    def __init__(self, dir: Path) -> None:
        self.dir = dir
        self.load_data_from_drive()
        
    def load_data_from_drive(self):
        print(f'loading data from {self.dir}...')
        with open(os.path.join(self.dir, 'train_all_features_dict*.pkl'), 'rb') as fn:
            self.train_data_dict = pickle.load(fn)

        with open(os.path.join(self.dir, 'test_all_features_dict*.pkl'), 'rb') as fn:
            self.test_data_dict = pickle.load(fn)

        with open(os.path.join(self.dir, 'train_all_labels_dict*.pkl'), 'rb') as fn:
            self.train_label_dict = pickle.load(fn)

        with open(os.path.join(self.dir, 'test_all_labels_dict*.pkl'), 'rb') as fn:
            self.test_label_dict = pickle.load(fn)
            
    def normalize_data_by_first_point(self, data_dict: dict) -> dict:
        normalize_dict = {}
        for key, value in data_dict.items():
            if len(value) == 0: continue
            normalize_dict[key] = np.array([data_array / np.mean(data_array, axis=0) for data_array in value])
        return normalize_dict
    
    def normalize_data_by_zscore(self, data_dict: dict) -> dict:
        normalize_dict = {}
        for key, value in data_dict.items():
            if len(value) == 0: continue
            normalize_dict[key] = np.array([(data_array - np.mean(data_array, axis=0)) / np.std(data_array, axis=0) for data_array in value])
        return normalize_dict

    def normalize_labels_by_mean(self, label_dict: dict) -> dict:
        normalize_dict = {}
        for key, value in label_dict.items():
            if len(value) == 0: continue
            normalize_dict[key] = np.array([label_array / np.mean(label_array) for label_array in value])
        return normalize_dict
    
    
    def process(self, augument=True, shuffle=True) -> Tuple[dict, dict, dict, dict]:
        if TFConsts.DataPreProcess.DATA_NORM_BY_FIRST_POINT:
            norm_train_data_dict = self.normalize_data_by_first_point(self.train_data_dict)
            norm_test_data_dict = self.normalize_data_by_first_point(self.test_data_dict)
            
        if TFConsts.DataPreProcess.DATA_NORM_BY_ZSCORE:
            norm_train_data_dict = self.normalize_data_by_zscore(self.train_data_dict)
            norm_test_data_dict = self.normalize_data_by_zscore(self.test_data_dict)
        
        if TFConsts.DataPreProcess.LABEL_NORM_BY_MEAN:
            norm_train_label_dict = self.normalize_labels_by_mean(self.train_label_dict)
            norm_test_label_dict = self.normalize_labels_by_mean(self.test_label_dict)
        
        train_data = np.array([j for i in norm_train_data_dict.keys() for j in norm_train_data_dict[i]])
        train_label = np.array([j for i in norm_train_label_dict.keys() for j in norm_train_label_dict[i]])
        print(f'train data shape {train_data.shape}')
        print(f'train label shape {train_label.shape}')
        
        test_data = np.array([j for i in norm_test_data_dict.keys() for j in norm_test_data_dict[i]])
        test_label = np.array([j for i in norm_test_label_dict.keys() for j in norm_test_label_dict[i]])
        print(f'test data shape {test_data.shape}')
        print(f'test label shape {test_label.shape}')
        
        if augument:
            flipped_train_data = np.flip(train_data, 2)
            flipped_train_label = np.flip(train_label, 1)
            train_data = np.concatenate([train_data, flipped_train_data], axis=0)
            train_label = np.concatenate([train_label, flipped_train_label], axis=0)
            print(f'augumented train data shape {train_data.shape}')
            print(f'augumented train label shape {train_label.shape}')
        
        if shuffle:
            np.random.seed(7)
            np.random.shuffle(train_data)

            np.random.seed(7)
            np.random.shuffle(train_label)
        
        return train_data, test_data, train_label, test_label