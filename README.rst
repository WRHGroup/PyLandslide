===========
Pylandslide
===========

Pylandslide is a machine learning-assisted open-source Python tool for landslide susceptibility mapping and uncertainty analysis.

Introduction
============

For further details on how to install and use the tool, please refer to the `Documentation <https://ierrg.github.io/PyLandslide/>`__.

Pylandslide is a tool for spatial mapping of landslide susceptibility. The tool uses “qualitative map combination,” in which the effects of multiple factors that contribute to landslide occurrence are combined using weights. The tool uses Machine learning to determine weights and their uncertainties. The tool is also designed to conduct sensitivity analysis based on weight ranges and spatially compare the outcomes of different weight choices.

Landslide susceptibility and uncertainty analysis can be performed in Pylandslide either through high-level commands or using Python code. Either way, some inputs to different methods and functionalities need to be provided through JSON-based document format. The `Documentation <https://ierrg.github.io/PyLandslide/>`__ page provides further details on how to use the tool and provide the required inputs.

Installation
============

Pylandslide should work on Python 3.6 (or later) on Windows, Linux or OS X.

See the documentation for information on `how to install Pylandslide <https://ierrg.github.io/PyLandslide/>`__.

Pylandslide can be installed by running:

.. code-block:: console

    pip install PyLandslide

For advanced users, developers, and those who with to contribute to further development of PyLandslide, make sue to have installed the required `dependencies <https://ierrg.github.io/PyLandslide/>`__. Then clone the repository:

.. code-block:: console

    git clone https://github.com/IERRG/PyLandslide.git

Once the repository is cloned, navigate to its directory and run:

.. code-block:: console

    python setup.py install

Or the following for development mode:

.. code-block:: console

    python setup.py develop

Citation
========

Please cite the following papers when using Pylandslide:


    1. Basheer, Oommen, Takamatsu & Suzuki. Machine learning and sensitivity analysis approach to quantify uncertainty in landslide susceptibility mapping. https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099356212142224352/idu1aab5df2016d2814a3c1bf5b11897f7cbd136 (2022).


License
=======

Copyright (C) 2023, `Mohammed Basheer <https://scholar.google.com/citations?user=KM_oVpkAAAAJ&hl=en>`__.


Pylandslide is released under the GNU General Public License.
