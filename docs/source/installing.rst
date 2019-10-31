==========
Installing
==========

If you use Anaconda or Miniconda, you can install psiexperiment from the bburan channel:

.. code-block:: shell

    conda install -c bburan psiexperiment

If you don't use Anaconda, psiexperiment can be installed via pip: 

.. code-block:: shell

    pip install psiexperiment

To install from source (e.g., if you plan to develop):

.. code-block:: shell

    git clone https://github.com/bburan/psiexperiment
    pip install -e ./psiexperiment

If you want to set up psiexperiment in development mode, but would rather pull in the dependencies via conda instead of pip:

.. code-block:: shell

    conda create -n <envname>
    conda install -c bburan -n <envname> --only-deps psiexperiment
    git clone https://github.com/bburan/psiexperiment
    pip install -e ./psiexperiment


Requirements
------------

The code is written with Python >= 3.6 in mind. The actual list of requirements are lengthy, but you may only need a subset depending on the plugins you use. The core requirements are:

* enaml
* numpy
* palettable
* scipy

Plugin-specific requirements:

* *BcolzStore* - bcolz, json_tricks
* *NIDAQEngine* - pydaqmx
* *TextStore* - json_tricks


After installing
----------------

First, create the required configuration file that contains keys indicating where various files (logging, data, calibration, settings, temporary data, preferences, layout, IO and experiments) are saved. The configuration file defaults to `~/psi/config.py`. The `PSI_CONFIG` environment variable can be used to override the default path.

where `<path>` is where you want the calibration, data and other configuration files should be stored:

.. code-block:: shell

    psi-config create --base-directory <path>

To view where the configuration file was saved:

.. code-block:: shell

    psi-config show

Now, open that file in your preferred Python editor and update the variables to point to where you want the various files stored. By default, you can have all files created by `psiexperiment` saved under a single `BASE_DIRECTORY`. Alternatively, you may want to be more specific (e.g., log files go here, data goes there, etc.). Feel free to customize as needed.

* *LOG_ROOT*: Location where log files are stored. These files are used for debugging.
* *DATA_ROOT*: Location where data files are stored. These files are generated when running an experiment and contain all data acquired by the experiment. 
* *CAL_ROOT*: Location where calibration data files are stored. These files are generated when running a calibration and are often required when running an experiment.
* *PREFERENCES_ROOT*: Location where experiment-specific preferences are stored.
* *LAYOUT_ROOT*: Location where experiment-specific layouts are stored.
* *IO_ROOT*: Location where system configuration is stored.

Once you have customized the configuration file, the folders can be created automatically if they don't already exist:

.. code-block:: shell

    psi-config create-folders

Now, go to where you defined `IO_ROOT` and create a file that ends with the extension `.enaml`. By convention, the file name should match the hostname of your computer (e.g., if your computer is called `bobcat`, then the file would be `bobcat.enaml`); however, this is not a requirement. To create a template that you can work with:

.. code-block:: shell

    psi-config create-io

Inside this file, you will describe the configuration of your system using Enaml_ syntax. 

.. _Enaml: http://enaml.readthedocs.io/en/latest/
