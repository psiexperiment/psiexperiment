========================
Using Psiexperiment
========================

This section provides instructions for end-users on how to launch and manage experiments using the ``psi`` command-line tool.

---------------------------
Launching an Experiment
---------------------------

Experiments (known as paradigms) are launched from the command line. The basic syntax is:

.. code-block:: bash

    psi <paradigm_name> --io <io_manifest_name>

*   **paradigm_name**: The unique identifier of the experiment definition (e.g., ``tone_pips``).
*   **io_manifest_name**: The name of your hardware configuration file (e.g., ``ni_daq_setup``).

Common Options
--------------
*   ``--preferences <file>``: Load a specific set of user preferences (e.g., parameter values) upon startup.
*   ``--debug``: Enable debug logging to the console.
*   ``--no-layout``: Start with a default window layout instead of loading the last saved layout.

---------------------------
The Main Interface
---------------------------

Once launched, the psiexperiment main window will appear. It consists of several key areas:

1.  **Toolbar**: Contains the "Start" and "Stop" buttons, as well as an "Apply" button for committing parameter changes mid-experiment.
2.  **Dock Panes**: Functional areas (e.g., plots, parameter editors) that can be moved or resized.
3.  **Status Bar**: Located at the bottom, showing the current experiment state and any active background tasks.

---------------------------
Managing Parameters
---------------------------

Parameters are organized into groups within the **Context** pane. 

*   **Editing**: You can click on most parameter values to edit them. Changes are staged until you click the **Apply** button on the toolbar.
*   **Mathematical Expressions**: Many fields support Python-style math (e.g., ``10 + 5``) or references to other parameters.
*   **Roving**: If a parameter is marked as "roving," it will follow the sequence defined in its associated selector.

---------------------------
Data Storage
---------------------------

By default, psiexperiment prompts you for a folder to save data when the experiment starts. 

*   **Data Format**: Data is typically saved in formats like HDF5, Zarr, or CSV, depending on which "sinks" are configured in the paradigm.
*   **Auto-Naming**: If configured, psiexperiment can automatically generate folder names based on the current date, subject ID, and experiment type.
