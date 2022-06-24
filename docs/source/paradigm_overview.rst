=========================
What makes an experiment?
=========================

An experiment is known as a *paradigm*. A paradigm is a collection of plug-in modules (some required, some optional) that interact with each other to run the experiment. The core paradigms that ship with psiexperiment can be found in ``psi.paradigms.descriptions``. Psiexperiment ships with a collection of paradigms for operant behavior (go-nogo), auditory testing (ABR, DPOAE, MEMR, EFR), noise exposure, and basic calibration of acoustic systems.

-------------------
What is a paradigm?
-------------------

A paradigm is described using the ``ParadigmDescription`` class. Here's an example description:

.. code-block:: enaml

    from psi.experiment.api import ParadigmDescription
    PATH = 'psi.paradigms.behavior.'
    CORE_PATH = 'psi.paradigms.core.'

    ParadigmDescription(
        'auto_gonogo', 'Auto GO-NOGO', 'animal', [
            {'manifest': PATH + 'behavior_auto_gonogo.BehaviorManifest'},
            {'manifest': PATH + 'behavior_mixins.BaseGoNogoMixin'},
            {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser'},
            {'manifest': CORE_PATH + 'video_mixins.PSIVideo'},
            {'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
             'attrs': {'fft_time_span': 1, 'fft_freq_lb': 5, 'fft_freq_ub': 24000, 'y_label': 'Level (dB)', 'source': 'microphone'}
             },
        ],
    )

The first argument, ``'auto_gonogo'`` is a unique identifier for the paradigm. To launch this experiment, you would type ``psi auto_gonogo`` at the command line. The fourth argument is a list of manifests that can be loaded as part of the experiment. Some of them, such as the ``BehaviorManifest``, are required. Others can be optionally loaded.  The manifests are referenced by the module name plus class name. To override defaults on the class, a dictionary can be passed via ``attrs``.

-------------------
What is a manifest?
--------------------

Briefly, a manifest defines how the associated plugin interacts with other plugins in the psiexperiment ecosystem. The manifest should subclass either ``PSIManifest`` or ``ExperimentManifest``. For most custom-written plugins, you will use ``ExperimentManifest``. Here's the definition for the ``SignalFFTViewManifest``:

.. code-block:: enaml

    from enaml.workbench.api import Extension

    from psi.core.enaml.api import ExperimentManifest
    from psi.data.plots import (ChannelPlot, FFTChannelPlot, FFTContainer,
                                TimeContainer, ViewBox)

    enamldef SignalFFTViewManifest(ExperimentManifest): manifest:

        id = 'signal_fft_view'
        name = 'signal_fft_view'
        title = 'Signal view (PSD)'

        # By re-defining attributes of the "children" in the graph hierarchy as
        # attributes of the top-level manifest, we can easily modify the values
        # of these attributes in the PluginDescription by passing in an attrs
        # dictionary.
        alias fft_time_span: fft_plot.time_span
        alias fft_freq_lb: fft_container.freq_lb
        alias fft_freq_ub: fft_container.freq_ub
        alias source_name: fft_plot.source_name
        alias y_label: fft_vb.y_label
        alias apply_calibration: fft_plot.apply_calibration
        alias waveform_averages: fft_plot.waveform_averages

        Extension:
            id = manifest.id  + '.plots'
            point = 'psi.data.plots'

            FFTContainer: fft_container:
                name << manifest.name + '_container'
                label << manifest.title
                freq_lb = 5
                freq_ub = 50000

                ViewBox: fft_vb:
                    name << manifest.name + '_vb'
                    y_min = -10
                    y_max = 100
                    y_mode = 'mouse'
                    save_limits = True

                    FFTChannelPlot: fft_plot:
                        name << manifest.name + '_plot'
                        source_name = 'microphone'
                        pen_color = 'k'
                        time_span = 0.25

This manifest extends the ``psi.data.plots`` extension point by adding a new FFT plot that will plot the running FFT of the signal source defined by ``source_name``. Another good example is the reward dispenser. This example demonstrates how you can define some basic functionality and then subclass the manifest to customize it:

.. code-block:: enaml

    enamldef BaseRewardDispenser(ExperimentManifest): manifest:

        attr duration
        attr output_name

        Extension:
            id = manifest.id + '.actions'
            point = 'psi.controller.actions'
            ExperimentAction:
                event = 'deliver_reward'
                command = f'{manifest.output_name}.fire'
                kwargs = {'duration': manifest.duration}

        Extension:
            id = manifest.id + '.status_item'
            point = 'psi.experiment.status'

            StatusItem:
                label = 'Total dispensed'
                Label:
                    text << str(workbench \
                                .get_plugin('psi.controller') \
                                .get_output(manifest.output_name) \
                                .total_fired)

        Extension:
            id = manifest.id + '.toolbar'
            point = 'psi.experiment.toolbar'
            rank = 2000
            Action:
                text = 'Dispense reward'
                triggered ::
                    controller = workbench.get_plugin('psi.controller')
                    controller.invoke_actions('deliver_reward')
                enabled <<  workbench.get_plugin('psi.controller').experiment_state \
                    not in ('initialized', 'stopped')


    enamldef WaterBolusDispenser(BaseRewardDispenser): manifest:

        id = 'water_bolus_dispenser'
        name = 'Water bolus dispenser'
        required = True

        duration = C.lookup('water_dispense_duration')
        output_name = 'water_dispense'

        Extension:
            id = manifest.id + '.parameters'
            point = 'psi.context.items'

            Parameter:
                name = 'water_dispense_duration'
                label = 'Water dispense trigger duration (s)'
                compact_label = 'D'
                default = 1
                scope = 'arbitrary'
                group_name = 'trial'

The first extension is to the ``psi.controller.actions`` point where we define a command that is called each time the ``deliver_reward`` event occurs. The ``deliver_reward`` event is generated by the core behavior controller whenever it determines that the subject has met the criteria for recieving a reward. By defining this as an event, we can link any number  of actions (defined by ``ExperimentAction``). Here, the action is to find the digital output defined by ``output_name`` and call the ``fire`` command with the specified trigger duration. In the LBHB system, this digital output is linked to a solenoid that opens when the trigger goes high, thereby allowing water to flow through into a lick spout that the subject has access to.  This is a good example of where you can easily customize what happens during the ``deliver_reward`` stage to switch to an alternate reward system by loading a different plugin. 

The second extension is to the ``psi.experiment.status`` point which is a user-facing panel that shows information about the experiment. This must subclass ``StatusItem`` and contribute a specific widget (in this case, a ``Label``, but other good widgets include a ``ProgressBar``). The extension to ``psi.experiment.toolbar`` adds a button that invokes all actions connected to the ``deliver_reward`` event. Finally, the ``psi.context.items`` extension manages all parameters (subclasses of ``ContextItems``). Parameters are variables (e.g., intertrial duration, reward trigger duration, number fo trials, etc.) that the user may want to control.
