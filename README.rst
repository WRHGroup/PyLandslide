===========
PyLandslide
===========

PyLandslide is a machine learning-assisted open-source Python tool for landslide susceptibility mapping and uncertainty analysis.

Introduction
============

For details on how to install and use the tool, please refer to the `Documentation <https://WRHGroup.github.io/PyLandslide/>`__.

PyLandslide is a tool for spatial mapping of landslide susceptibility. The tool uses "qualitative map combination," in which the effects of multiple factors that contribute to landslide occurrence are combined using weights. The tool uses Machine learning to determine weights and their uncertainties. The tool is also designed to conduct sensitivity analysis based on weight ranges and spatially compare the outcomes of different weight choices.

Landslide susceptibility and uncertainty analysis can be performed in PyLandslide either through high-level commands or using Python code. Either way, some inputs to different methods and functionalities need to be provided through JSON-based document format. The `Documentation <https://WRHGroup.github.io/PyLandslide/>`__ page provides further details on how to use the tool and provide the required inputs.

.. image:: https://raw.githubusercontent.com/WRHGroup/PyLandslide/main/docs/figs/frm.jpg
   :width: 750px

Installation
============

PyLandslide works on Python 3.6 (or later) on Windows, Linux, or OS X.

See the documentation for details on `how to install PyLandslide <https://WRHGroup.github.io/PyLandslide/installation.html>`__.

PyLandslide can be installed by running:

.. code-block:: console

    pip install PyLandslide

For advanced users, developers, and those who wish to contribute to the development of PyLandslide, make sure to have installed the required `dependencies <https://WRHGroup.github.io/PyLandslide/installation.html>`__. Then clone the repository:

.. code-block:: console

    git clone https://github.com/WRHGroup/PyLandslide.git

Once the repository is cloned, navigate to its directory and run:

.. code-block:: console

    python setup.py install

Or the following for development mode:

.. code-block:: console

    python setup.py develop

Citation
========

Please cite the following papers when using PyLandslide:


    1. Basheer, M., Oommen, T., 2024. PyLandslide: A Python tool for landslide susceptibility mapping and uncertainty analysis. Environmental Modelling and Software. 106055. https://doi.org/10.1016/j.envsoft.2024.106055.
    2. Basheer, M., Oommen, T., Takamatsu, M., Suzuki, S., 2022. Machine learning and sensitivity analysis approach to quantify uncertainty in landslide susceptibility mapping, Policy Research Working Paper. Washington, D.C. https://doi.org/10.1596/1813-9450-10264.


License
=======

Copyright (C) 2023, `Mohammed Basheer <https://scholar.google.com/citations?user=KM_oVpkAAAAJ&hl=en>`__ and `Thomas Oommen <https://scholar.google.com/citations?user=EP89cqIAAAAJ&hl=en>`__.


PyLandslide is released under the GNU General Public License.
