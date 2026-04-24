==========
Installing
==========

Psiexperiment is a Python-based framework. It is recommended to install it within a virtual environment (such as ``conda`` or ``venv``) to manage dependencies effectively.

Supported Hardware
------------------

Psiexperiment provides drivers and interfaces for a variety of hardware platforms:

* **National Instruments**: Supported via NIDAQmx drivers (requires ``pyDAQmx``).
* **Tucker-Davis Technologies (TDT)**: Supported via ActiveX drivers (requires ``tdtpy``).
* **Sound Cards**: Basic support for system sound cards via ``sounddevice``.
* **Biosemi**: Supported via a custom interface.

Installation
------------

The easiest way to install psiexperiment is via ``pip``:

.. code-block:: bash

    pip install psiexperiment

If you need support for specific hardware, you can install the optional dependencies:

.. code-block:: bash

    # For National Instruments support
    pip install psiexperiment[ni]

    # For TDT support
    pip install psiexperiment[tdt]

    # For standard soundcard support
    pip install psiexperiment[soundcard]

Dependencies
------------

Psiexperiment requires Python >= 3.7. The core dependencies include:

* **enaml**: The declarative UI and plugin framework.
* **numpy** and **scipy**: For signal processing and math.
* **pandas**: For data management.
* **pyqtgraph**: For high-performance real-time plotting.
* **psidata** and **psiaudio**: Core libraries for data I/O and auditory signals.

Configuring Psiexperiment
-------------------------

Before running your first experiment, you must create a configuration file. This file tells psiexperiment where to store logs, data, calibration files, and user preferences.

1. Create the configuration
...........................

Use the ``psi-config`` tool to initialize your environment. In the example below, ``PATH`` is the root directory where all experiment-related files will be stored.

.. code-block:: bash

    psi-config create --base-directory PATH

For example:

.. code-block:: bash

    psi-config create --base-directory C:/psiexperiment_data

2. Standard Hardware Templates
..............................

Psiexperiment ships with several standard hardware templates. You can use these as a starting point for your own IO manifest:

* **Biosemi32** / **Biosemi64**: Biosemi EEG systems.
* **Medusa4ZTDT** / **RA4PATDT**: TDT RZ6 configurations.
* **PXIe-1062**: National Instruments PXIe chassis with PXI-4461 cards.

You can create a skeleton IO manifest based on one of these templates:

.. code-block:: bash

    psi-config create-io PXIe-1062

This will create an ``.enaml`` file in your ``IO_ROOT`` that you can then customize.

3. Finalizing Folders
.....................

Once your configuration is created, ensure all required subdirectories exist:

.. code-block:: bash

    psi-config create-folders

4. Verifying the Configuration
..............................

To see the current location of your configuration file and its settings:

.. code-block:: bash

    psi-config show

You can open the generated ``config.py`` file in any text editor to fine-tune the paths for ``LOG_ROOT``, ``DATA_ROOT``, ``CAL_ROOT``, etc.
