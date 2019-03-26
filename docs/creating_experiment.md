# Designing a new experiment

## Getting started

Psiexperiment is a plugin-based system where the plugins determine the experiment workflow. At a minimum, an experiment must provide the following:

* A list of parameters and/or results

* The inputs and outputs that will be used

* The stimuli (i.e., tokens) that will be generated

* Actions to take when certain events occur.
 
Your experiment configuration file will typically contain the following entries:

    enamldef ControllerManifest(BaseManifest): manifest:

        Extension:
            id = EXPERIMENT + '.sinks'
            point = 'psi.data.sinks'

        Extension:
            id = EXPERIMENT + '.tokens'
            point = 'psi.token.tokens'

        Extension:
            id = EXPERIMENT + '.io'
            point = 'psi.controller.io'

        Extension:
            point = 'psi.context.selectors'

        Extension:
            id = EXPERIMENT + '.context'
            point = 'psi.context.items'

        Extension:
            id = EXPERIMENT + '.actions'
            point = 'psi.controller.actions'

## Actions

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

## Creating your own custom plugins

When defining your own subclasses of `PSIManifest`, we recommend the following naming convetions to minimize name collisions:

    Extension:
        id = manifest.id + '.commands'
        point = 'enaml.workbench.core.commands'

        Command:
            id = contribution.name + '.do_action'
            ...

All subclasses of `PSIManifest` have access to the attached `contribution` as an attribute.
