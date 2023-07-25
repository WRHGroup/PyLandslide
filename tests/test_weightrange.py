import os
import datetime
import pytest
import pandas as pd
import numpy as np

from PyLandslide.weightrange import WeightRangeEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def test_json_loading():
    #Test that the json file loaded
    WeightRangeModel = WeightRangeEstimator('tests/files/weight_range_json_file.json')
    WeightRangeModel.setup()

    features_file_path = os.path.join(os.getcwd(),'tests/files/ml/features.csv')
    assert(WeightRangeModel.features_file == features_file_path)
    targets_file_path = os.path.join(os.getcwd(),'tests/files/ml/targets.csv')
    assert(WeightRangeModel.targets_file == targets_file_path)
    tree_depths_file_path = os.path.join(os.getcwd(),'tests/files/ml/tree_depths_by_target.csv')
    assert(WeightRangeModel.tree_depths_file == tree_depths_file_path)

def test_csv_df_loading():
    #Test that the json file loaded
    WeightRangeModel = WeightRangeEstimator('tests/files/weight_range_json_file.json')
    WeightRangeModel.setup()
    
    targets_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.targets_file, index_column = 'id')
    features_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.features_file, index_column = 'id')
    tree_depths_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.tree_depths_file, index_column = 'target')

    assert(type(targets_df) == pd.DataFrame)
    assert(type(features_df) == pd.DataFrame)
    assert(type(tree_depths_df) == pd.DataFrame)

def test_overall_accuracy_metric_model():
    WeightRangeModel = WeightRangeEstimator('tests/files/weight_range_json_file.json')
    WeightRangeModel.setup()
    
    targets_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.targets_file, index_column = 'id')
    features_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.features_file, index_column = 'id')
    tree_depths_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.tree_depths_file, index_column = 'target')
    
    target_names = targets_df.columns

    X_train, X_test, Y_train, Y_test = train_test_split(features_df, targets_df, test_size = WeightRangeModel.size_testing_sample)

    #create a RandomForestClassifier model
    importances = WeightRangeModel.feature_importance_model(targets=Y_train[target_names[0]], features=X_train, max_tree_depth=tree_depths_df.at[target_names[0],"tree_depth"], n_estimators=WeightRangeModel.number_trees, cores=WeightRangeModel.cores)

    metrics = WeightRangeModel.overall_accuracy(importances[1],X_train,Y_train[target_names[0]],X_test,Y_test[target_names[0]])
    assert(type(metrics) == tuple)
    assert(len(metrics) == 2)
    assert(type(metrics[0]) == np.float64)
    assert(type(metrics[1]) == np.float64)
    assert(metrics[0] <= 1)
    assert(metrics[1] <= 1)

def test_random_forest_model_metric_model():
    WeightRangeModel = WeightRangeEstimator('tests/files/weight_range_json_file.json')
    WeightRangeModel.setup()
    
    targets_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.targets_file, index_column = 'id')
    features_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.features_file, index_column = 'id')
    tree_depths_df = WeightRangeModel.load_data_from_csv(WeightRangeModel.tree_depths_file, index_column = 'target')
    
    target_names = targets_df.columns

    X_train, X_test, Y_train, Y_test = train_test_split(features_df, targets_df, test_size = WeightRangeModel.size_testing_sample)

    #create a RandomForestClassifier model
    importances = WeightRangeModel.feature_importance_model(targets=Y_train[target_names[0]], features=X_train, max_tree_depth=tree_depths_df.at[target_names[0],"tree_depth"], n_estimators=WeightRangeModel.number_trees, cores=WeightRangeModel.cores)
    
    assert(type(importances) == tuple)
    assert(len(importances) == 2)
    assert(type(importances[1]) == RandomForestClassifier)
    assert(type(importances[0]) == np.ndarray)
    assert(max(importances[0]) <= 1)
