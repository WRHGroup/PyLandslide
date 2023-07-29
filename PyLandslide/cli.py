#from .uncertainty import Model
import click
import os
import sys
import logging
from PyLandslide.weightrange import *
from PyLandslide.sensitivity import *
from PyLandslide.weightrange_preparation import *

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass

@cli.command(name='mldata')
@click.option('-f', '--file-name', type=str, default="weight_range_data_preparation.json")
def mldata(file_name):
    logger.info('Starting the process.')
    WeightRangePreparationModel = WeightRangePreparation(json_file=file_name)
    WeightRangePreparationModel.setup()
    WeightRangePreparationModel.extract()

@cli.command(name='weightrange')
@click.option('-f', '--file-name', type=str, default="weight_range_json_file.json")
def weightrange(file_name):
    logger.info('Starting the process.')
    WeightRangeModel = WeightRangeEstimator(json_file=file_name)
    WeightRangeModel.setup()
    WeightRangeModel.calculate_weight_range()

@cli.command(name='sensitivity')
@click.option('-f', '--file-name', type=str, default="sensitivity_json_file.json")
@click.option('-t', '--trials', type=int, default=10)
def sensitivity(file_name, trials):
    logger.info('Starting the process.')
    SensitivityModel = SensitivityEstimator(json_file=file_name, trials=trials)
    SensitivityModel.setup()
    SensitivityModel.execute()

@cli.command(name='generate')
@click.option('-f', '--file-name', type=str, default="sensitivity_json_file.json")
@click.option('-c', '--csv-sensitivity', type=str, default="sensitivity_results.csv")
@click.option('-i', '--index', type=int, default=1)
def generate(file_name, csv_sensitivity, index):
    logger.info('Starting the process.')
    SensitivityModel = SensitivityEstimator(json_file=file_name)
    SensitivityModel.setup()
    SensitivityModel.generate(index = index, csv_sensitivity=csv_sensitivity)

@cli.command(name='compare')
@click.option('-f', '--file-name', type=str, default="sensitivity_json_file.json")
@click.option('-l1', '--layer1', type=str, default="susceptibility_0.tif")
@click.option('-l2', '--layer2', type=str, default="susceptibility_1.tif")
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
