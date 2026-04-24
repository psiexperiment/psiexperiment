==========================
Designing a new experiment
==========================

Designing a new experiment in psiexperiment is the process of defining how different core plugins and your custom logic interact through hooks. This page will guide you through the process of building a new *paradigm* (experiment definition) from scratch.

Core plugins
------------
Every experiment relies on five core plugins that are always loaded:

* **Context** (``psi.context``): Manages parameters, trial selectors, and mathematical expressions.
* **Controller** (``psi.controller``): The central state machine. It manages hardware I/O and links events to actions.
* **Data** (``psi.data``): Handles real-time data flow, visualization (plots), and persistent storage (sinks).
* **Experiment** (``psi.experiment``): Manages the main workspace layout, toolbars, and global metadata.
* **Token** (``psi.token``): Provides signal primitives (e.g., tone pips, noise) for generating waveforms.

Step 1: Defining the Experiment Manifest
-----------------------------------------

Your experiment starts with an Enaml manifest that subclasses ``ExperimentManifest``. This manifest will declare extensions to the core plugins to customize their behavior for your experiment.

.. code-block:: enaml

    from enaml.workbench.api import Extension
    from psi.core.enaml.api import ExperimentManifest
    from psi.controller.api import ExperimentAction
    from psi.context.api import Parameter, Result

    enamldef MyExperimentManifest(ExperimentManifest): manifest:
        id = 'my_experiment'

        # Contribute actions to the controller
        Extension:
            id = manifest.id + '.actions'
            point = 'psi.controller.actions'

            # Initialize the context when the "Start" button is clicked
            ExperimentAction:
                event = 'experiment_initialize'
                command = 'psi.context.initialize'
                kwargs = {'selector': 'default', 'cycles': 1}

            # Define custom actions for experimental events
            ExperimentAction:
                event = 'trial_start'
                command = 'my_experiment.prepare_stimulus'

        # Contribute parameters to the context
        Extension:
            id = manifest.id + '.parameters'
            point = 'psi.context.items'

            Parameter:
                name = 'stimulus_frequency'
                label = 'Stimulus Frequency (Hz)'
                default = 1000.0
                group_name = 'stimulus'

            Result:
                name = 'reaction_time'
                label = 'Reaction Time (s)'

        # Contribute plots to the data plugin
        Extension:
            id = manifest.id + '.plots'
            point = 'psi.data.plots'
            # (Add Plot definitions here)

Experimental Lifecycle and Actions
----------------------------------

Psiexperiment uses an event-driven system to manage the flow of an experiment. You connect **Events** (e.g., ``trial_start``) to **Commands** (e.g., ``psi.context.initialize``) using an ``ExperimentAction``.

Common Lifecycle Events:
.......................
* `plugins_started`: Fires once all plugin manifests are loaded. Use this for global logging or one-time discovery.
* `experiment_initialize`: The first event when the "Start" button is pressed. This is where you should always call ``psi.context.initialize``.
* `context_initialized`: Fires after the context plugin is ready. This is where hardware I/O is finalized (via ``psi.controller.finalize_io``).
* `experiment_prepare`: This is the stage where hardware engines are configured. Use this for setup that requires the context values to be available.
* `engines_configured`: Hardware is ready to acquire/generate data.
* `experiment_start`: The final event before the hardware acquisition loop begins.
* `experiment_end`: Triggered when the experiment stops (either via a button click or a programmatic completion event).

The Power of Actions
....................
Actions allow you to insert your own code or invoke existing commands at any point. By defining custom events (e.g., ``subject_licked``) in your hardware I/O manifest, you can trigger complex responses (e.g., ``deliver_reward``) without hardcoding the connection between the lick spout and the reward dispenser.

Input and Output (I/O)
----------------------

Hardware is defined in a separate :doc:`IO manifest <io_manifest>`. In your experiment manifest, you then connect these hardware channels to processing blocks (e.g., ``ContinuousOutput`` or ``ExtractEpochs``).

.. code-block:: enaml

    Extension:
        id = manifest.id + '.io'
        point = 'psi.controller.io'

        ContinuousOutput: ao:
            name = 'stimulus_output'
            # Use a parameter defined in the context via 'C'
            source = 'tone_pip'
            target_name = C.output_channel

The ``C`` object is a manifest-level variable that provides a shorthand for looking up values from the context plugin (e.g., ``C.output_channel``).

Creating Custom Plugins
-----------------------

When building large experiments, we recommend breaking functionality into small, reusable plugins (subclasses of ``ExperimentManifest``). This allows you to mix and match features (like different reward types or stimulus paradigms) easily.

Common Gotchas
--------------
* **Initialization**: You must always call ``psi.context.initialize`` during the ``experiment_initialize`` event. This is not automatic because some experiments may need to perform custom initialization steps before the context is ready.
* **Active Channels**: Outputs and inputs are only configured by the hardware engine if they are "active". A channel is active if it is connected to a plot, a data sink, or an explicit action.
* **Coordinate Naming**: Always use globally unique names for parameters and I/O channels to avoid collisions when multiple plugins are loaded simultaneously.
* **Asynchronous Calls**: If your plugin needs to update the GUI from a background thread (e.g., a hardware callback), you must use ``deferred_call`` to schedule the update on the main UI thread.
