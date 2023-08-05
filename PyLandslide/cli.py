#from .uncertainty import Model
import click
import os
import sys
import logging
from PyLandslide.weightrange import *
from PyLandslide.sensitivity import *
from PyLandslide.data_preparation import *

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass

@cli.command(name='mldata', help="Prepare features and targets data for ML.")
@click.option('-f', '--file-name', type=str, default="1_weight_range_data_preparation.json", help="JSON file containing the data preparation config.")
def mldata(file_name):
    logger.info('Starting the process.')
    WeightRangePreparationModel = DataPreparation(json_file=file_name)
    WeightRangePreparationModel.setup()
    WeightRangePreparationModel.extract()

@cli.command(name='weightrange', help="Determine weight range based on ML.")
@click.option('-f', '--file-name', type=str, default="2_weight_range_json_file.json", help="JSON file containing the weight range config.")
def weightrange(file_name):
    logger.info('Starting the process.')
    WeightRangeModel = WeightRangeEstimator(json_file=file_name)
    WeightRangeModel.setup()
    WeightRangeModel.calculate_weight_range()

@cli.command(name='coregister', help="Align raster data before using them.")
@click.option('-f', '--folder-name', type=str, default="raster_data", help="Folder containing the raster data.")
def coregister(folder_name):
    logger.info('Starting the process.')
    CoRegisterModel = DataPreparation(folder_name=folder_name)
    CoRegisterModel.adjust()
    CoRegisterModel.align()

@cli.command(name='sensitivity', help="Sensitivity of landslide hazard to weight uncertainty.")
@click.option('-f', '--file-name', type=str, default="3_sensitivity_json_file_historical_rainfall.json", help="JSON file containing the sensitivity config.")
@click.option('-t', '--trials', type=int, default=10, help="Number of sensitivity trials.")
def sensitivity(file_name, trials):
    logger.info('Starting the process.')
    SensitivityModel = SensitivityEstimator(json_file=file_name, trials=trials)
    SensitivityModel.setup()
    SensitivityModel.execute()

@cli.command(name='generate', help="Generate a landslide hazard layer based on a sensitivity trial.")
@click.option('-f', '--file-name', type=str, default="3_sensitivity_json_file_historical_rainfall.json", help="JSON file containing the sensitivity config.")
@click.option('-c', '--csv-sensitivity', type=str, default="sensitivity_results.csv", help="CSV file of the sensitivity results.")
@click.option('-i', '--index', type=int, default=1, help="Index of the desired trail.")
def generate(file_name, csv_sensitivity, index):
    logger.info('Starting the process.')
    SensitivityModel = SensitivityEstimator(json_file=file_name)
    SensitivityModel.setup()
    SensitivityModel.generate(index = index, csv_sensitivity=csv_sensitivity)

@cli.command(name='compare', help="Compare two landslide hazard raster file.")
@click.option('-f', '--file-name', type=str, default="3_sensitivity_json_file_historical_rainfall.json", help="JSON file containing the sensitivity config.")
@click.option('-l1', '--layer1', type=str, default="susceptibility_0.tif", help="First landslide hazard raster file.")
@click.option('-l2', '--layer2', type=str, default="susceptibility_1.tif", help="Second landslide hazard raster file.")
def compare(file_name, layer1, layer2):
    logger.info('Starting the process.')
    SensitivityModel = SensitivityEstimator(json_file=file_name)
    SensitivityModel.setup()
    SensitivityModel.compare(layer1=layer1, layer2=layer2)

def start_cli():
    """ Start the command line interface. """
    from . import logger
    import sys
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(ch)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    cli(obj={})
