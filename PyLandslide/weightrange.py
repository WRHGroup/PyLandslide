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
from sklearn.ensemble import RandomForestClassifier
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

        if loaded_file.get('output_file') is None:
            print('output file set to weight_ranges.csv in the cwd. If you wish to change this default value, add number_trees to the inputs provided in the JSON file')
            self.output_file = os.path.join(os.getcwd(), 'weight_ranges.csv')
        else:
            self.output_file = os.path.join(os.getcwd(), loaded_file.pop('output_file'))

        if loaded_file.get('size_testing_sample') is None:
            print('size of testing sample has been set to 0.2. If you wish to change this default value, add size_testing_sample to the inputs provided in the JSON file')
            self.size_testing_sample = 0.2
        else:
            self.size_testing_sample = loaded_file.pop('size_testing_sample')

        if loaded_file.get('number_of_iterations') is None:
            print('number of iterations has been set to 500. If you wish to change this default value, add number_of_iterations to the inputs provided in the JSON file')
            self.number_of_iterations = 50
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

    def load_data_from_csv(self, filename, index_column):
        temp = pd.read_csv(filename)
        input_data = temp.set_index(index_column)
        return input_data

    def calculate_weight_range(self):
        targets_df = self.load_data_from_csv(self.targets_file, index_column = 'id')
        features_df = self.load_data_from_csv(self.features_file, index_column = 'id')
        tree_depths_df = self.load_data_from_csv(self.tree_depths_file, index_column = 'target')
        
        #Cach target and feature names
        target_names = targets_df.columns
        feature_names = features_df.columns
        all_columns = list(feature_names)
        all_columns.append("target")
        all_columns.append("training_overall_accuracy")
        all_columns.append("testing_overall_accuracy")

        #create an empty df based on the number of targets and features under consideration
        results = pd.DataFrame(index = list(range(int(self.number_of_iterations))), columns = all_columns)
        results.index.name = 'iteration'

        for itera in range(int(self.number_of_iterations)):
            #Note: This randomly split the data in 80% train and 20% test data
            X_train, X_test, Y_train, Y_test = train_test_split(features_df, targets_df, test_size = self.size_testing_sample)
            #Loop through targets
            for t, targ in enumerate(target_names):
                print('-------------------------------------')
                print('Iteration number:',t+1+(itera*len(target_names)))

                #create a RandomForestClassifier model
                importances = self.feature_importance_model(targets=Y_train[targ], features=X_train, max_tree_depth=tree_depths_df.at[targ,"tree_depth"], n_estimators=self.number_trees, cores=self.cores)

                metrics = self.overall_accuracy(importances[1],X_train,Y_train[targ],X_test,Y_test[targ])

                #save the values of relative importance into the 'results' dataframe
                if metrics[0] >= self.performance_cutoff and metrics[1] >= self.performance_cutoff:
                    results.at[itera, "target"] = targ
                    results.at[itera, "training_overall_accuracy"] = metrics[0]
                    results.at[itera, "testing_overall_accuracy"] = metrics[1]
                    for f, feat in enumerate (feature_names):
                        results.at[itera, feat] = importances[0][f]

        results.dropna(how='all').to_csv(self.output_file)

    def feature_importance_model(self, targets, features, max_tree_depth, n_estimators, cores):
        #create a RandomForestClassifier 
        model = RandomForestClassifier(random_state=1, max_depth=max_tree_depth, n_estimators = n_estimators, n_jobs = cores)
        model.fit(features,targets)
        #calculate relative importance
        importance = model.feature_importances_
        return importance, model
        
    def overall_accuracy(self, mod,X_train,Y_train,X_test,Y_test):
        Y_predicted_test = mod.predict(X_test)
        Y_predicted_train = mod.predict(X_train)

        matrix_test = confusion_matrix(Y_test, Y_predicted_test)
        matrix_train = confusion_matrix(Y_train, Y_predicted_train)

        TP_TN_test = 0
        FP_fn_test = 0
        for y in range(len(matrix_test)):
            for x in range(len(matrix_test)):
                if y == x:
                    TP_TN_test += matrix_test[y][x]
                else:
                    FP_fn_test += matrix_test[y][x]

        TP_TN_train = 0
        FP_fn_train = 0
        for y in range(len(matrix_train)):
            for x in range(len(matrix_train)):
                if y == x:
                    TP_TN_train += matrix_train[y][x]
                else:
                    FP_fn_train += matrix_train[y][x]

        overall_accuracy_test = TP_TN_test/(TP_TN_test+FP_fn_test)
        overall_accuracy_train = TP_TN_train/(TP_TN_train+FP_fn_train)
        
        print('overall accuracy in training =', round(overall_accuracy_train,3))
        print('overall accuracy in testing =', round(overall_accuracy_test,3))

        return overall_accuracy_train, overall_accuracy_test

    def setup(self):
        self.load_data_from_json()
        logger.info('Setting up WeightRangeEstimator based on the file: "{}"'.format(self.json_file))
