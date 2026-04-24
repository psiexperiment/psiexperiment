=======================================
Example: A Simple Auditory Experiment
=======================================

To help you understand how all the pieces of psiexperiment fit together, let's walk through the creation of a simple **Tone-in-Noise** detection experiment. In this experiment:
1.  A continuous background noise is played.
2.  At random intervals, a brief tone pip is presented.
3.  The user can adjust the frequency and level of the tone pip via the GUI.
4.  Data (both the raw signal and experimental events) is saved to disk.

---------------------------
1. The IO Manifest
---------------------------

First, we define our hardware configuration in ``my_io.enaml``. We'll assume a National Instruments DAQ with one output for the speaker and one input for a monitoring microphone.

.. code-block:: enaml

    from psi.controller.api import IOManifest, NIEngine, AnalogOutput, AnalogInput

    enamldef MyIO(IOManifest):
        id = 'my_hardware'
        NIEngine:
            name = 'NI-DAQmx'
            dev_name = 'Dev1'
            master_clock = True

            AnalogOutput: speaker:
                name = 'speaker'
                channel = 'ao0'
                fs = 100000.0

            AnalogInput: microphone:
                name = 'microphone'
                channel = 'ai0'
                fs = 100000.0

---------------------------
2. The Experiment Manifest
---------------------------

Now, we define the logic in ``tone_in_noise.enaml``. We'll contribute to several core plugins.

Defining the Stimuli
--------------------
We contribute to ``psi.token.tokens`` to define our noise and tone.

.. code-block:: enaml

    Extension:
        point = 'psi.token.tokens'
        # Continuous background noise
        Noise:
            name = 'background_noise'
        # Brief tone pip
        Token:
            name = 'tone_pip'
            Tone:
                freq = C.tone_frequency
                level = C.tone_level
            CosineEnvelope:
                duration = 0.05
                rise_time = 0.005

Connecting to Hardware
----------------------
We contribute to ``psi.controller.io`` to wire these signals to our speaker.

.. code-block:: enaml

    Extension:
        point = 'psi.controller.io'
        # Continuous background masker
        ContinuousOutput: masker:
            name = 'masker_output'
            source = 'background_noise'
            target_name = 'speaker'
        # On-demand target tone
        EpochOutput: target:
            name = 'target_output'
            source = 'tone_pip'
            target_name = 'speaker'

Defining Parameters
-------------------
We contribute to ``psi.context.items`` to expose settings to the GUI.

.. code-block:: enaml

    Extension:
        point = 'psi.context.items'
        Parameter:
            name = 'tone_frequency'
            label = 'Frequency (Hz)'
            default = 4000.0
        Parameter:
            name = 'tone_level'
            label = 'Level (dB SPL)'
            default = 60.0

Controlling the Flow
--------------------
We use ``psi.controller.actions`` to automate the experiment.

.. code-block:: enaml

    Extension:
        point = 'psi.controller.actions'
        # Initialize context on start
        ExperimentAction:
            event = 'experiment_initialize'
            command = 'psi.context.initialize'
        # Play a tone when a trial starts
        ExperimentAction:
            event = 'trial_start'
            command = 'target_output.start'

---------------------------
3. The Paradigm Description
---------------------------

Finally, we bundle these into a paradigm in ``descriptions.py``.

.. code-block:: python

    from psi.experiment.api import ParadigmDescription

    ParadigmDescription(
        'tone_noise', 'Tone-in-Noise Experiment', 'auditory', [
            {'manifest': 'tone_in_noise.MyExperimentManifest'},
        ],
    )

---------------------------
How It Works Together
---------------------------

1.  **Launch**: Run ``psi tone_noise --io my_hardware``.
2.  **Initialize**: When you click **Start**, the ``experiment_initialize`` event fires, calling ``psi.context.initialize``.
3.  **IO Setup**: The controller plugin sees the ``ContinuousOutput`` and immediately starts streaming the ``background_noise`` to the speaker.
4.  **Running**: When your code (or a timer) fires the ``trial_start`` event, the ``target_output.start`` command is called, which pulls a 50ms tone from the ``tone_pip`` generator and mixes it into the speaker stream.
5.  **Data**: Any configured ``Sink`` (in ``psi.data.sinks``) will see the events and data streams and write them to the data folder you selected.

This simple example demonstrates how psiexperiment handles the complex task of multi-signal mixing, hardware synchronization, and parameter management through a few lines of declarative code.
