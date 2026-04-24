========================
Plugin & Hook Reference
========================

Psiexperiment is built on a modular plugin architecture. This page provides a comprehensive reference of all core plugins, their available hooks (extension points), and the types of contributions they accept.

------------------------------------------
psi.context (Parameters and Sequences)
------------------------------------------

The context plugin manages all experimental variables, their values, and how they change over time.

**Extension Point: psi.context.items**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to declare parameters, results, and other context-related items.

- **Supports Factory**: Yes
- **Contribution Types**:
    - **Parameter**: A user-configurable variable.
        - `name` (str): Unique identifier.
        - `label` (str): Human-readable label for the GUI.
        - `default` (any): Initial value.
        - `dtype` (str): NumPy-compatible type (e.g., 'float64', 'int32').
        - `group_name` (str): Optional group identifier for UI organization.
        - `scope` (str): Determines how the parameter is saved ('trial', 'session', etc.).
    - **Result**: A value calculated by a plugin after an event occurs.
        - `name`, `label`, `dtype`, `group_name`.
    - **ContextGroup**: Groups related parameters in a single UI pane.
        - `name`, `label`, `visible`, `hide_when`.
    - **ContextSet**: A set of parameters that should be roved together.
    - **ContextMeta**: Metadata about context items (e.g., whether they are linked).
    - **Expression**: A mathematical formula assigning a value to a parameter.
        - `parameter` (str): Target parameter name.
        - `expression` (str): Formula (e.g., ``"f1_level + 10"``).

**Extension Point: psi.context.selectors**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to add trial sequence generators.

- **Supports Factory**: No
- **Contribution Types**:
    - **SequenceSelector**: Iterates through a fixed list of parameter combinations.
        - `name` (str): Unique identifier (often ``'default'``).
        - `order` (str): ``'sequential'`` or ``'random'``.
    - **RandomSelector**: Randomizes parameter values based on a distribution.

**Extension Point: psi.context.symbols**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to add functions or constants to the expression evaluator.

- **Supports Factory**: No
- **Contribution Types**:
    - **Function**: Exposes a Python function.
        - `name` (str): Name available in expressions (e.g., ``"db"``).
        - `function` (callable): Python function reference.
    - **ImportedSymbol**: Imports a module or object from a module.
        - `name`, `module`, `obj`.

------------------------------------------
psi.controller (Hardware and Flow)
------------------------------------------

The controller plugin manages hardware I/O and the high-level experiment state.

**Extension Point: psi.controller.io**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to define the hardware configuration.

- **Supports Factory**: No
- **Contribution Types**:
    - **Engine**: Hardware drivers.
        - `name` (str): Unique identifier.
        - `master_clock` (bool): If True, this engine drives the system timing.
        - `weight` (int): Initialization order.
        - **Subclasses**: ``NIEngine``, ``LabJackEngine``, ``SoundcardEngine``, ``BiosemiEngine``.
    - **Channel**: Hardware-specific I/O lines.
        - `name` (str): Unique identifier.
        - `channel` (str): Hardware address (e.g., ``"ao0"``).
        - `fs` (float): Sampling frequency.
        - **Subclasses**: ``AnalogInput``, ``AnalogOutput``, ``DigitalInput``, ``DigitalOutput``.
    - **Input**: Processing blocks for incoming data.
        - `name` (str): Identifier.
        - `source_name` (str): Source channel or processing block.
        - **Subclasses**: ``Discard``, ``ExtractEpochs``, ``Average``, ``IIRFilter``, ``Threshold``.
    - **Output**: Processing blocks for outgoing signals.
        - `name` (str): Identifier.
        - `target_name` (str): Target hardware channel.
        - `source` (str): Name of the token/signal generator.
        - **Subclasses**: ``ContinuousOutput``, ``EpochOutput``, ``QueuedEpochOutput``.

**Extension Point: psi.controller.actions**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to link experimental events to commands.

- **Supports Factory**: No
- **Contribution Types**:
    - **ExperimentAction**: Maps an event to a command.
        - `event` (str): The event identifier.
        - `command` (str): The command ID to invoke.
        - `kwargs` (dict): Optional arguments for the command.
        - `weight` (int): Execution priority (lower is earlier).
        - `delay` (float): Delay in seconds before invocation.
    - **ExperimentEvent**: Declares a new custom event.
    - **ExperimentState**: Declares a state (e.g., ``"running"``) that generates events.
    - **ExperimentCallback**: Maps an event to a Python callback.
    - **EventLogger**: Logs event occurrences.

**Extension Point: psi.controller.wrapup**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used for final cleanup or displaying results after the experiment stops.

- **Supports Factory**: Yes (**REQUIRED**)
- **Usage**: The factory must return a callable that accepts the workbench as its first argument.

------------------------------------------
psi.data (Storage and Visualization)
------------------------------------------

Handles real-time plotting and data persistence.

**Extension Point: psi.data.sinks**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to add data storage backends.

- **Supports Factory**: No
- **Contribution Types**:
    - **Sink**: Base class for storage backends.
        - `name` (str): Unique identifier.
        - **Subclasses**: ``ZarrStore``, ``TextStore``, ``CSVStore``, ``HDF5Store``, ``EventLog``.

**Extension Point: psi.data.plots**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to contribute real-time visualizations.

- **Supports Factory**: Yes
- **Contribution Types**:
    - **BasePlotContainer**: Layout element for grouping plots.
        - **Subclasses**: ``FFTContainer``, ``TimeContainer``.
    - **ViewBox**: A coordinate system for plots.
        - `y_min`, `y_max`, `y_mode` ('auto', 'fixed', 'mouse').
    - **BasePlot**: Individual plot types.
        - `source_name` (str): Source channel or processing block.
        - **Subclasses**: ``FFTChannelPlot``, ``ChannelPlot``, ``HistogramPlot``.

------------------------------------------
psi.experiment (UI and Metadata)
------------------------------------------

Manages the application workspace and global metadata.

**Extension Point: psi.experiment.workspace**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to add new panes to the main application window.

- **Supports Factory**: Yes
- **Contribution Types**:
    - **DockItem**: A dockable UI panel.
        - `name` (str): Unique identifier.
        - `title` (str): Window title.

**Extension Point: psi.experiment.status**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to add display elements to the status bar.

- **Supports Factory**: Yes
- **Contribution Types**:
    - **StatusItem**: A small display widget.
        - `label` (str): Display label.

**Extension Point: psi.experiment.toolbar**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to add controls to the application toolbar.

- **Supports Factory**: No
- **Contribution Types**:
    - **Action**: A toolbar button.
        - `text` (str): Button text.
        - `triggered` (declarative): Logic to run when clicked.

**Extension Point: psi.experiment.preferences**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Registers plugin attributes for persistence.

- **Supports Factory**: Yes
- **Contribution Types**:
    - **Preferences**: Defines which attributes should be saved.
    - **PluginPreferences**: Saves all preference-tagged members of a plugin.
    - **ItemPreferences**: Saves specific items.

**Extension Point: psi.experiment.metadata**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Records global information about the experimental session.

- **Supports Factory**: Yes
- **Contribution Types**:
    - **MetadataItem**: A key-value pair.
        - `name` (str): Unique identifier.
        - `value` (any): Initial or static value.

------------------------------------------
psi.token (Signal Primitives)
------------------------------------------

Provides building blocks for creating auditory stimuli.

**Extension Point: psi.token.tokens**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to add new signal primitives.

- **Supports Factory**: Yes
- **Contribution Types**:
    - **ContinuousBlock**: Infinite signal generators.
        - **Subclasses**: ``Noise``, ``CosSum``.
    - **EpochBlock**: Finite signal generators.
        - **Subclasses**: ``Tone``, ``Chirp``, ``Click``, ``Silence``.

------------------------------------------
psi.controller.calibration.channels
------------------------------------------

Manages hardware calibration routines.

**Extension Point: psi.controller.calibration.channels**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used to add calibration algorithms.

- **Supports Factory**: No
- **Contribution Types**:
    - **BaseCalibrate**: Defines how to calibrate a hardware output.
        - `outputs` (dict): Dictionary mapping output names to required parameters.
        - **Subclasses**: ``ToneCalibrate``, ``ChirpCalibrate``, ``GolayCalibrate``, ``ClickCalibrate``.
