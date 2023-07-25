#loading neccessery packages
import os
import click
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from osgeo import gdal
import random
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, max_error, mean_squared_error, confusion_matrix
logger = logging.getLogger(__name__)

class WeightRangeEstimator(object):
    def __init__(self, json_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_file = json_file

    def load_data_from_json(self, **kwargs):
        """Load data from a file
        """
        data = os.path.join(os.getcwd(), self.json_file)
        if isinstance(data, str):
            logging.info('Loading data from file: "{}"'.format(data))
            with open(data, "r") as f:
                loaded_file = json.load(f)

        if loaded_file.get('features_file') is None:
            raise ValueError('features_file has not been found in the JSON file')
        self.features_file = os.path.join(os.getcwd(), loaded_file.pop('features_file'))

        if loaded_file.get('targets_file') is None:
            raise ValueError('targets_file has not been found in the JSON file')
        self.targets_file = os.path.join(os.getcwd(), loaded_file.pop('targets_file'))

        if loaded_file.get('tree_depths_file') is None:
            raise ValueError('tree_depths_file has not been found in the JSON file')
        self.tree_depths_file = os.path.join(os.getcwd(), loaded_file.pop('tree_depths_file'))

        if loaded_file.get('number_trees') is None:
            print('number of tree has been set to 100. If you wish to change this default value, add number_trees to the inputs provided in the JSON file')
            self.number_trees = 100
        else:
            self.number_trees = loaded_file.pop('number_trees')

        if loaded_file.get('size_testing_sample') is None:
            print('size of testing sample has been set to 0.2. If you wish to change this default value, add size_testing_sample to the inputs provided in the JSON file')
            self.size_testing_sample = 0.2
        else:
            self.size_testing_sample = loaded_file.pop('size_testing_sample')

        if loaded_file.get('number_of_iterations') is None:
            print('number of iterations has been set to 500. If you wish to change this default value, add number_of_iterations to the inputs provided in the JSON file')
            self.number_of_iterations = 500
        else:
            self.number_of_iterations = loaded_file.pop('number_of_iterations')

        if loaded_file.get('cores') is None:
            print('number of cores has been set to 1. If you wish to change this default value, add cores to the inputs provided in the JSON file')
            self.cores = 1
        else:
            self.cores = loaded_file.pop('cores')

        if loaded_file.get('performance_cutoff') is None:
            print('performance cutoff has been set to 0.75. If you wish to change this default value, add performance_cutoff to the inputs provided in the JSON file')
            self.performance_cutoff = 0.75
        else:
            self.performance_cutoff = loaded_file.pop('performance_cutoff')

    def setup(self):
        self.load_data_from_json()
        logger.info('Setting up WeightRangeEstimator based on the file: "{}"'.format(self.json_file))
