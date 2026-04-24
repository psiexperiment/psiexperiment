=========================
What makes an experiment?
=========================

In psiexperiment, an experiment is known as a *paradigm*. A paradigm is not a single large piece of code but rather a collection of functional plug-in modules (some required, some optional) that interact with each other to realize the experimental design. 

This modular approach ensures that core features like data storage, hardware communication, and parameter management can be reused across vastly different experimental types. For a detailed look at the underlying plugin architecture, see the :doc:`architecture` documentation.

-------------------
What is a paradigm?
-------------------

A paradigm is defined using the ``ParadigmDescription`` class. It serves as a manifest of all the plugins that should be loaded together to run the experiment.

Here's an example of how a paradigm is described:

.. code-block:: enaml

    from psi.experiment.api import ParadigmDescription
    
    CORE_PATH = 'psi.paradigms.core.'
    MY_PATH = 'my_lab_plugins.'

    ParadigmDescription(
        # Unique identifier for the paradigm
        'tone_pips', 
        # Human-readable name
        'Tone Pip Experiment', 
        # Experiment type (e.g., 'animal' or 'calibration')
        'auditory', 
        [
            # Required core plugins
            {'manifest': MY_PATH + 'tone_pip_controller.TonePipManifest'},
            
            # Optional visual monitoring plugin
            {'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
             'selected': True,
             'attrs': {
                 'fft_time_span': 1.0, 
                 'source': 'microphone',
                 'y_label': 'Level (dB)'
              }
            },
            
            # Additional hardware drivers or monitoring tools
            {'manifest': CORE_PATH + 'video_mixins.PSIVideo', 'selected': False},
        ],
    )

Key Concepts of Paradigm Descriptions:
-------------------------------------
*   **Unique ID**: The first argument (e.g., ``'tone_pips'``) is how you launch the experiment via the command line: ``psi tone_pips``.
*   **Plugin List**: The list contains one or more plugin manifests (identified by their full module path).
*   **Attrs**: You can override the default values of attributes in a manifest by passing an ``attrs`` dictionary. This is a powerful way to customize a generic plugin (like an FFT view) for a specific experiment without writing new code.
*   **Selected vs Required**: Plugins can be marked as ``selected`` (loaded by default but can be unchecked in the launcher) or ``required`` (always loaded).

-------------------
What is a manifest?
-------------------

A *manifest* is the Enaml declaration that defines how a plugin connects to the rest of the psiexperiment ecosystem. Most plugins will subclass ``ExperimentManifest``.

Here is a simplified example of the ``SignalFFTViewManifest``, which contributes a real-time plot to the experiment:

.. code-block:: enaml

    from enaml.workbench.api import Extension
    from psi.core.enaml.api import ExperimentManifest
    from psi.data.plots import (FFTChannelPlot, FFTContainer, ViewBox)

    enamldef SignalFFTViewManifest(ExperimentManifest): manifest:

        id = 'signal_fft_view'
        name = 'signal_fft_view'
        title = 'Signal view (PSD)'

        # Attributes that can be overridden via ParadigmDescription 'attrs'
        alias fft_time_span: fft_plot.time_span
        alias source_name: fft_plot.source_name

        Extension:
            id = manifest.id  + '.plots'
            point = 'psi.data.plots'

            FFTContainer: fft_container:
                label << manifest.title

                ViewBox: fft_vb:
                    y_min = -10
                    y_max = 100

                    FFTChannelPlot: fft_plot:
                        source_name = 'microphone'
                        time_span = 0.25

In this example, the manifest uses an **Extension** to contribute a new plot to the ``psi.data.plots`` hook. By defining ``alias`` properties, it makes internal settings (like ``time_span``) accessible from the top-level manifest, allowing them to be customized in the ``ParadigmDescription``.

Another common use for manifests is defining **Actions** that respond to experimental events:

.. code-block:: enaml

    enamldef RewardPlugin(ExperimentManifest): manifest:

        id = 'reward_dispenser'
        
        attr reward_duration = 0.5

        Extension:
            id = manifest.id + '.actions'
            point = 'psi.controller.actions'
            
            ExperimentAction:
                # Respond to a behavioral event
                event = 'subject_responded_correctly'
                # Invoke a command on a hardware output
                command = 'water_valve.fire'
                kwargs = {'duration': manifest.reward_duration}

By using this declarative approach, psiexperiment allows you to build complex, reactive systems by simply stating which plugins should be used and how they should be configured.
