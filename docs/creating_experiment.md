# Designing a new experiment

## Getting started

Psiexperiment is a plugin-based system where the plugins determine the experiment workflow. At a minimum, an experiment must provide the following:

* A list of parameters and/or results

* The inputs and outputs that will be used

* The stimuli (i.e., tokens) that will be generated

* Actions to take when certain events occur.
 
Your experiment configuration file will define `EXPERIMENT`, which is the name of the experiment and will typically contain the following extensions:

    enamldef ControllerManifest(BaseManifest): manifest:

        Extension:
            id = EXPERIMENT + '.sinks'
            point = 'psi.data.sinks'
            ...

        Extension:
            id = EXPERIMENT + '.tokens'
            point = 'psi.token.tokens'
            ...

        Extension:
            id = EXPERIMENT + '.io'
            point = 'psi.controller.io'
            ...

        Extension:
            id = EXPERIMENT + '.selectors'
            point = 'psi.context.selectors'
            ...

        Extension:
            id = EXPERIMENT + '.context'
            point = 'psi.context.items'
            ...

        Extension:
            id = EXPERIMENT + '.actions'
            point = 'psi.controller.actions'
            ...

Let's take a closer look at each of the extensions.

### Actions

At a minimum, you will typically define the following two actions (customized for your needs):

    Extension:
        id = EXPERIMENT + '.actions'
        point = 'psi.controller.actions'

        ExperimentAction:
            event = 'experiment_initialize'
            command = 'psi.context.initialize'
            kwargs = {'selector': 'default', 'cycles': 1}

        ExperimentAction:
            event = 'engines_configured'
            command = 'dpoae.start'
            kwargs = {'delay': 0.5}

When you press the `start` button on the toolbar, this fires a sequence of three events, `experiment_initialize`, `experiment_prepare` and `experiment_start`.  While `experiment_initialize` and `experiment_prepare` are very similar, certain actions may require that the context has been initialized. To simplify this, we have added `experiment_initialize`. The `psi.context.initialize` command should *always* be bound to this event (if it's bound to `experiment_prepare`, you may get errors if commands bound to `experiment_prepare` need to get the current value of a variable).

Under the hood, the controller will configure the engines during the `experiment_prepare` phase. If you want to configure one of the outputs (in this case, `dpoae`) during this phase, be sure to bind it to the `engine_configured` event to ensure it gets executed after the engine is configured (the engine must be configured before it can properly receive waveform samples from the outputs).

#### Sequence of events during an experiment

* `plugins_started` - All plugins have finished loading. Now, you can perform actions that may require access to another plugin; however, do not assume that the plugins have finished initializing.

* `experiment_initialize` - TODO

* `context_initialized` - TODO

* `experiment_prepare` - TODO

* `engines_configured` - TODO

* `experiment_start` - TODO

#### The power of actions

The actions allow you to insert your own code or invoke commands at any point in the experiment process. A few examples:

* The `abr_base.enaml` file calls a custom function when the `experiment_prepare` event is called. This function reviews the settings specified by the user to determine the sequence of the tone pips (e.g., conventional vs. interleaved, alternating polarity, etc.) and sets up the queue accordingly. While it's theoretically possible to set this using plugins offered by psiexperiment (e.g., alternating polarity could be specified as a "roving" context item), this custom function makes the user interface much simpler and more fool-proof.

* The `pistonphone_calibration.enaml` file calls a custom function, `calculate_sens` once the experiment is complete to calculate the sensitivity of the microphone. Note that the callback for the custom function is defined inside the extension to the `psi.controller.io` point.


### Input/Output

        Extension:
            id = EXPERIMENT + '.io'
            point = 'psi.controller.io'

            Blocked: hw_ai:
                duration = 0.1
                name = 'hw_ai'
                source_name = C.input_channel
                source ::
                    # Once the channel is linked
                    channel.start_trigger = ''
                    channel.samples = round(C.sample_duration * channel.fs)
                    channel.input_gain = C.input_gain

`C` is a controller manifest-level variable that allows for lookup of values defined via the context.


### Creating your own custom plugins

When defining your own subclasses of `PSIManifest`, we recommend the following naming convetions to minimize name collisions:

    Extension:
        id = manifest.id + '.commands'
        point = 'enaml.workbench.core.commands'

        Command:
            id = contribution.name + '.do_action'
            ...

All subclasses of `PSIManifest` have access to the attached `contribution` as an attribute.
