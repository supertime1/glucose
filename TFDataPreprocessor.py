""" 
This document includes functions to prepare training for tensorflow in cloud
"""

import numpy as np
from pathlib import Path
import os
import pickle
from typing import Tuple


class TFDataPreprocessor:
    def __init__(self, dir: Path) -> None:
        self.dir = dir
        self.load_data_from_drive()
        
    def load_data_from_drive(self):
        print(f'loading data from {self.dir}...')
        with open(os.path.join(self.dir, 'train_all_features_dict.pkl'),'rb') as fn:
            self.train_data_dict = pickle.load(fn)

        with open(os.path.join(self.dir, 'test_all_features_dict.pkl'),'rb') as fn:
            self.test_data_dict = pickle.load(fn)

        with open(os.path.join(self.dir, 'train_all_labels_dict.pkl'),'rb') as fn:
            self.train_label_dict = pickle.load(fn)

        with open(os.path.join(self.dir, 'test_all_labels_dict.pkl'),'rb') as fn:
            self.test_label_dict = pickle.load(fn)
            
    def normalize_data_by_first_point(self, data_dict: dict) -> dict:
        normalize_dict = {}
        for key, value in data_dict.items():
            if len(value) == 0: continue
            normalize_dict[key] = np.array([data_array / np.mean(data_array, axis=0) for data_array in value])
        return normalize_dict
    
    def standardize_data_by_zscore(self, data_dict: dict) -> dict:
        standardize_dict = {}
        for key, value in data_dict.items():
            if len(value) == 0: continue
            standardize_dict[key] = np.array([(data_array - np.mean(data_array, axis=0)) / np.std(data_array, axis=0) for data_array in value])
        return standardize_dict

    def standardize_labels_by_mean(label_dict: dict) -> dict:
        standardize_dict = {}
        for key, value in label_dict.items():
            if len(value) == 0: continue
            standardize_dict[key] = np.array([label_array / np.mean(label_array) for label_array in value])
        return standardize_dict
    
    def process(self) -> Tuple[dict, dict, dict, dict]:
        pass