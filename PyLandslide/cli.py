#from .uncertainty import Model
import click
import os
import sys
import logging
from PyLandslide.weightrange import *
import random

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option('-f', '--file-name', type=str, default="json_file.json")
def weightrange(file_name):
    logger.info('Starting the process.')
    WeightRangeModel = WeightRangeEstimator(json_file = file_name)
    WeightRangeModel.setup()
    WeightRangeModel.calculate_weight_range()

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
