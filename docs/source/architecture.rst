============================
Psiexperiment Architecture
============================

Psiexperiment is a modular, plugin-based framework for experimental control and data acquisition. It leverages the `Enaml Workbench <https://enaml.readthedocs.io/en/latest/api_ref/workbench/index.html>`_ to provide a flexible architecture where experiments are composed of reusable functional modules.

Core Concepts
=============

Paradigm
--------
An experiment in psiexperiment is called a *paradigm*. A paradigm is defined by a :class:`~psi.experiment.paradigm_description.ParadigmDescription`, which lists the set of plugins (manifests) required to run the experiment.

Manifest
--------
A *manifest* (subclass of :class:`~psi.core.enaml.manifest.ExperimentManifest`) defines a plugin's behavior and how it interacts with other plugins. It declares:
*   **Extension Points (Hooks)**: Areas where other plugins can contribute functionality.
*   **Extensions**: Contributions to hooks provided by other plugins.
*   **Commands**: Functions that can be invoked via the workbench.

Workbench
---------
The :class:`~psi.experiment.workbench.PSIWorkbench` is the central hub. It registers core plugins and ensures they are properly initialized and "bound" to each other. Every manifest has access to the ``context``, ``controller``, and ``data`` plugins via its ``manifest`` object.


Key Plugins
===========

Psiexperiment relies on five core plugins that establish the framework for any experiment.

psi.context (Parameters and Sequences)
--------------------------------------
Manages all experimental variables and their values. It handles roving parameters, trial sequences, and mathematical expressions.

*   **Hook: selectors**: Used to add new trial sequence generators (e.g., random, blocked, or custom behavioral selectors).
*   **Hook: items**: Used to add parameters, results, or roving parameters to the experiment.
*   **Hook: symbols**: Used to add mathematical functions or constants available for use in parameter expressions (e.g., ``np.sin``, ``db()``).

psi.controller (Hardware and Flow)
----------------------------------
The "engine room" of the experiment. It manages hardware I/O and the high-level state machine.

*   **Hook: io**: Used to define hardware engines (e.g., NI-DAQmx, LabJack) and their associated input/output channels.
*   **Hook: actions**: Used to register :class:`~psi.controller.experiment_action.ExperimentAction` objects that link events (e.g., ``trial_start``) to commands (e.g., ``deliver_reward``).
*   **Hook: wrapup**: Used to add tasks that run after the experiment ends (e.g., saving data, showing a summary window).

psi.data (Storage and Visualization)
------------------------------------
Handles the flow of acquired data to storage and real-time displays.

*   **Hook: sinks**: Used to add data storage backends (e.g., HDF5, CSV, Zarr).
*   **Hook: plots**: Used to contribute real-time visualizations such as FFTs, time-series plots, or histograms.

psi.experiment (UI and Metadata)
--------------------------------
Manages the main application window and global experiment information.

*   **Hook: workspace**: Used to add new dockable panes (``DockItem``) to the main window.
*   **Hook: status**: Used to add items to the status bar at the bottom of the window.
*   **Hook: toolbar**: Used to add buttons and controls to the application toolbar.
*   **Hook: metadata**: Used to record experiment-wide metadata (e.g., software versions, hardware serial numbers).
*   **Hook: preferences**: Used to register preferences that should be saved and loaded between sessions.

psi.token (Signal Primitives)
-----------------------------
Provides a system for defining signal generation and processing blocks.

*   **Hook: tokens**: Used to add new signal primitives (e.g., tone pips, noise bursts, chirps) that can be used in stimulus definitions.


How Plugins Work Together
=========================

The power of psiexperiment comes from how these plugins interact via hooks. For example, a behavioral plugin might:
1.  Contribute a **Parameter** to ``psi.context.items`` to define the reward duration.
2.  Contribute an **ExperimentAction** to ``psi.controller.actions`` to trigger a reward when a lick occurs.
3.  Contribute a **StatusItem** to ``psi.experiment.status`` to show the total number of rewards delivered.
4.  Contribute a **DockItem** to ``psi.experiment.workspace`` to provide a custom UI for monitoring the subject's performance.


Experiment Lifecycle and Event Sequence
======================================

When an experiment is started, the controller plugin manages a specific sequence of events to ensure all components are initialized and synchronized correctly.

Initialization Phase
--------------------
1.  **plugins_started**: All plugins have loaded and are ready for registration.
2.  **experiment_initialize**: The user clicks the "Start" button. This event should trigger ``psi.context.initialize``.
3.  **context_initialized**: Fires after the context has successfully initialized its parameters and selectors.
4.  **io_configured**: Fires after the controller has connected all inputs and outputs (via ``finalize_io``).
5.  **experiment_prepare**: The main preparation phase. The controller handles ``psi.controller.configure_engines`` here.
6.  **engines_configured**: Fires after all hardware engines (NI-DAQmx, etc.) have been configured.

Running Phase
-------------
1.  **experiment_start**: The final event before data acquisition begins. Triggers ``psi.controller.start_engines``.
2.  **engines_started**: Fires once hardware is actively acquiring/generating data.
3.  **running**: The experiment is now in the ``running`` state.
4.  **trial_start / trial_end**: (If applicable) Events fired by specific behavioral or stimulus plugins.

Termination Phase
-----------------
1.  **experiment_end**: Triggered when the experiment finishes or the user clicks "Stop". Calls ``psi.controller.stop_engines``.
2.  **engines_stopped**: Fires once hardware has successfully stopped.
3.  **wrapup**: Tasks that run after engines stop (e.g., showing results).


Contributing to a Hook
======================

To extend psiexperiment, you create a new manifest and add an ``Extension`` block targeting the desired hook.

Example: Adding a Status Item
-----------------------------

.. code-block:: enaml

    from enaml.workbench.api import Extension
    from psi.core.enaml.api import ExperimentManifest
    from psi.experiment.api import StatusItem
    from enaml.widgets.api import Label

    enamldef MyStatusManifest(ExperimentManifest): manifest:
        id = 'my_status_plugin'

        Extension:
            id = manifest.id + '.status'
            point = 'psi.experiment.status'
            StatusItem:
                Label:
                    text << f"Status: {controller.experiment_state}"

Example: Adding an Action
-------------------------

.. code-block:: enaml

    from enaml.workbench.api import Extension
    from psi.core.enaml.api import ExperimentManifest
    from psi.controller.api import ExperimentAction

    enamldef MyActionManifest(ExperimentManifest): manifest:
        id = 'my_action_plugin'

        Extension:
            id = manifest.id + '.actions'
            point = 'psi.controller.actions'
            ExperimentAction:
                event = 'trial_start'
                command = 'my_plugin.do_something'

Creating Your Own Hook
----------------------

If you are developing a new core plugin, you can define your own hooks using the ``ExtensionPoint`` tag in your manifest. Other plugins can then contribute to it using the patterns shown above.
