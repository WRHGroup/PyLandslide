#from .uncertainty import Model
import click
import os
import sys
import logging
from PyLandside import *
import random

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option('-f', '--file-name', type=str, default="json_file.json")
def process(file_name):
    logger.info('Starting the process.')
    raster_calculations = Model(json_file = file_name)
    raster_calculations.setup()
    raster_calculations.run()

@cli.command()
@click.option('-f', '--file-name', type=str, default="json_file.json")
def design(file_name):
    logger.info('Starting the process.')
    raster_calculations = Model(json_file = file_name)
    raster_calculations.setup()
    raster_calculations.run()


@cli.command()
@click.option('-f', '--file-name', type=str, default="json_file.json")
@click.option('-c', '--csvfile-name', type=str, default="sorted_metrics.csv")
@click.option('-i', '--sol-index', type=int, default=1)
def dem(file_name, csvfile_name, sol_index):
    logger.info('Starting the process.')
    dem_creator = Model(json_file = file_name, csvfile_name = csvfile_name, sol_index = sol_index)
    dem_creator.setup()
    dem_creator.create_dem()


@cli.command()
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('--use-mpi/--no-use-mpi', default=False)
@click.option('-s', '--seed', type=int, default=None)
@click.option('-p', '--num-cpus', type=int, default=None)
@click.option('-n', '--max-nfe', type=int, default=1000)
@click.option('--pop-size', type=int, default=50)
@click.option('-a', '--algorithm', type=click.Choice(['NSGAII', 'NSGAIII', 'EpsMOEA']), default='NSGAII')
@click.option('-e', '--epsilons', multiple=True, type=float, default=(0.05, ))
@click.option('--divisions-outer', type=int, default=12)
@click.option('--divisions-inner', type=int, default=0)
def search(filename, use_mpi, seed, num_cpus, max_nfe, pop_size, algorithm, epsilons, divisions_outer, divisions_inner):
    import platypus
    #from platypus.mpipool import MPIPool

    logger.info('Loading model from file: "{}"'.format(filename))
    directory, model_name = os.path.split(filename)
    output_directory = os.path.join(directory, 'outputs')

    if algorithm == 'NSGAII':
        algorithm_klass = platypus.NSGAII
        algorithm_kwargs = {'population_size': pop_size}
    elif algorithm == 'NSGAIII':
        algorithm_klass = platypus.NSGAIII
        algorithm_kwargs = {'divisions_outer': divisions_outer, 'divisions_inner':divisions_inner}
    elif algorithm == 'EpsMOEA':
        algorithm_klass = platypus.EpsMOEA
        algorithm_kwargs = {'population_size': pop_size, 'epsilons': epsilons}
    else:
        raise RuntimeError('Algorithm "{}" not supported.'.format(algorithm))

    if seed is None:
        seed = random.randrange(sys.maxsize)

    search_data = {'algorithm': algorithm, 'seed': seed, 'user_metadata':algorithm_kwargs}

    wrapper = SaveNondominatedSolutionsArchive(filename, search_data=search_data, output_directory=output_directory, model_name=model_name)
    

    if seed is not None:
        random.seed(seed)

    logger.info('Starting model search.')

    #if use_mpi:
    #    pool = MPIPool()
    #    evaluator_klass = platypus.PoolEvaluator
    #    evaluator_args = (pool,)

    #    if not pool.is_master():
    #        pool.wait()
    #        sys.exit(0)

    #elif num_cpus is None:
    if num_cpus is None:
        evaluator_klass = platypus.MapEvaluator
        evaluator_args = ()
    else:
        evaluator_klass = platypus.ProcessPoolEvaluator
        evaluator_args = (num_cpus,)

    with evaluator_klass(*evaluator_args) as evaluator:
        algorithm = algorithm_klass(wrapper.problem, evaluator=evaluator, **algorithm_kwargs)
        
        algorithm.run(max_nfe, callback=wrapper.save_nondominant)


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
