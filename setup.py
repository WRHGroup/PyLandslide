from setuptools import setup, find_packages

setup(
    name='PyLandslide',
    version='1.0',
    description='Tools for landslide hazard uncertainty analysis.',
    url='',
    author='Mohammed Basheer',
    author_email='mohammedadamabbaker@gmail.com',
    packages=find_packages(),
    package_data={
        'PyLandslide': ['json/*.json'],
    },
    entry_points={
        'console_scripts': ['PyLandslide=PyLandslide.cli:start_cli'],
    }
)
