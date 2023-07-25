import os
import datetime
import pytest
import pandas
from numpy.testing import assert_allclose
from PyLandside.weightrange import WeightRangeEstimator


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
