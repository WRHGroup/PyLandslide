from setuptools import setup, find_packages

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='PyLandslide',
    version='0.0.6',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='Tools for landslide hazard uncertainty analysis.',
    url='https://github.com/IERRG/PyLandslide',
    author='Mohammed Basheer',
    author_email='mohammedadamabbaker@gmail.com',
    license='GNU',
    install_requires=['click','pandas','numpy','scikit-learn','geopandas','rasterio'],
    packages=find_packages(),
    package_data={
        'PyLandslide': ['json/*.json'],
    },
    entry_points={
        'console_scripts': ['PyLandslide=PyLandslide.cli:start_cli'],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ]
)
