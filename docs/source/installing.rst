==========
Installing
==========

Supported Hardware
------------------

Psiexperiment currently supports the following hardware:

* National Instruments (via the NIDAQmx drivers and pyDAQmx)
* TDT RZ6 (via the ActiveX drivers and tdtpy).
* Biosemi (via a custom port of the pyactivetwo library)

Using conda
-----------

If you use Anaconda or Miniconda, you can install psiexperiment from the psiexperiment channel. ``ENV`` is the name of the conda environment you wish to create to host psiexperiment. This is important because it allows you to have multiple versions of different Python packages (e.g., Numpy, Matplotlib, PyQtGraph, etc. without affecting psiexperiment).

.. code-block:: doscon

    conda create -n ENV -c psiexperiment psiexperiment

For example, if you want to call the environment `psi`, you would type:

.. code-block:: doscon

    conda create -n psi -c psiexperiment psiexperiment

If you plan to work on the source code, refer to the :doc:`instructions for developers<development>`.


Dependencies
------------

The code is written with Python >= 3.7 in mind. The actual list of requirements are lengthy, but you may only need a subset depending on the plugins you use. The core requirements are:

* enaml
* numpy
* palettable
* scipy
* pyqtgraph
* pandas

Plugin-specific requirements:

* **ZarrStore** or **BinaryStore** - zarr
* **BcolzStore** - bcolz
* **NIDAQEngine** - pydaqmx
* **TDTEngine** - tdtpy
* **BiosemiEngine** - pyactivetwo (customized port)


Configuring psiexperiment
-------------------------

First, create the required configuration file that contains information indicating where various files (logging, data, calibration, settings, temporary data, preferences, layout, IO and experiments) are saved. The configuration file defaults to ``~/psi/config.py``. The ``PSI_CONFIG`` environment variable can be used to override the default path.

Open an Anaconda prompt. Be sure to activate the enviornment to which you installed psiexperiment, e.g., assuming that the enviornment was named ``psi``:

.. code-block:: shell

    conda activate psi


Standard hadware configurations
...............................

Let's create the configuration. Psiexperiment ships with several standard hardware configurations that are commonly available in research laboratories:

* **Medusa4ZTDT** - TDT RZ6 with a Medusa4Z connected.
* **RA4PATDT** - TDT RZ6 with a RA4PA connected.
* **PXIe-1032** - NI PXIe with a PXI-4461 card.
* **Biosemi32** - Biosemi with 32 channels (requires ActiView to be configured properly).
* **Biosemi64** - Biosemi with 64 channels (requires ActiView to be configured properly).

If one of these standard IO configurations matches your equipment, then the rest of the set-up process is relatively simple. In the command below, ``PATH`` is where you want the calibration, data and other configuration files should be stored. ``IO`` is the hardware configuration for your system (the bolded text from the list above).

.. code-block:: shell

    psi-config create --base-directory PATH --io IO

For example, if you want to save data to ``c:/data`` and you have the TDT with the Medusa4Z:

.. code-block:: shell

    psi-config create --base-directory c:/data --io Medusa4ZTDT

To view where the configuration file was saved:

.. code-block:: shell

    psi-config show

Now, open that file in your preferred Python editor (Idle is fine as it's installed by default with Python) and update the variables to point to where you want the various files stored. By default, you can have all files created by psiexperiment saved under a single ``BASE_DIRECTORY``. Alternatively, you may want to be more specific (e.g., log files go here, data goes there, etc.). Feel free to customize as needed.

* **LOG_ROOT**: Location where log files are stored. These files are used for debugging.
* **DATA_ROOT**: Location where data files are stored. These files are generated when running an experiment and contain all data acquired by the experiment. 
* **CAL_ROOT**: Location where calibration data files are stored. These files are generated when running a calibration and are often required when running an experiment.
* **PREFERENCES_ROOT**: Location where experiment-specific preferences are stored.
* **LAYOUT_ROOT**: Location where experiment-specific layouts are stored.
* **IO_ROOT**: Location where system configuration is stored.
* **STANDARD_IO**: List of hardware configurations (see above). Usually there will be only one, but you may sometimes want to allow the user to select from several options (e.g., if you have both a RA4PA and Medusa4Z).

Once you have customized the configuration file, the folders can be created automatically if they don't already exist:

.. code-block:: shell

    psi-config create-folders

Nonstandard hardware configurations
...................................
First, create a skeleton file for your hardware configuration. ``SKELETON`` is the name of the template you want to base the configuration on:

.. code-block:: shell

    psi-config create-io SKELETON

Inside this file, you will describe the configuration of your system using Enaml_ syntax. This is known as the :doc:`input-output manifest<io_manifest>`

.. _Enaml: http://enaml.readthedocs.io/en/latest/
