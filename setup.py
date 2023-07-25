from setuptools import setup, find_packages

setup(
    name='PyLandside',
    version='1.0',
    description='Tools for landslide hazard uncertainty analysis.',
    url='',
    author='Mohammed Basheer',
    author_email='mohammedadamabbaker@gmail.com',
    packages=find_packages(),
    package_data={
        'PyLandside': ['json/*.json'],
    },
    entry_points={
        'console_scripts': ['PyLandside=PyLandside.cli:start_cli'],
    }
)
