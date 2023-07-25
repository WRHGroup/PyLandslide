import os
import datetime
import pytest
import pandas
from numpy.testing import assert_allclose

from landlev.model import Model


'''
def test_file_name():
    #Test that the directory is successfully loaded
    # load a model from a dem file
    model = Model('tests/files/json_file.json')
    model.setup()
    file_path = os.path.join(os.getcwd(),'tests/files/dem/1_m_dem.tif')
    assert(model.dem_file == file_path)

def test_setup_model():
    #Test that the loads the files successfully
    # load a model from a dem file
    model = Model('tests/files/json_file.json')
    model.setup()
    assert model.dataset is not None
    assert model.DEM_np is not None

def test_run_model_sinks():
    #Test that the model determines the sinks successfuly
    # load a model from a dem file
    model = Model('tests/files/json_file.json')
    model.setup()
    model.run()
    assert(model.number_of_sinks == 32969.0)
'''
