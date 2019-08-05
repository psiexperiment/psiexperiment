=======
Context
=======

Every experiment has a set of variables that define the behavior. These variables range from the stimulus frequency and level to the intertrial interval duration. Sometimes these variables need to be expressed as functions of other variables, or the value of the variable needs to vary in a random fashion.

A *context item* provides information about a value that is managed by the context plugin. When defining a context item in one of your plugin manifests, you will provide basic information about the item (e.g., a GUI label, compact GUI label and numpy dtype). This information will be used by plugins that interact with the context plugin (for example, the name and dtype of the context item will be used by the TextStore plugin to set up the table that stores acquired trial data).

There are currently three specific types of context items. A *result* is a value provided by a plugin. It cannot be defined by the user. Common use cases may include the values computed after a trial is acquired (e.g., one can compute the `reaction_time` and provide it as a result).

A *parameter* is a value that can be configured by the user before and during an experiment. While the value of the parameter can be modified by the user during the experiment, it cannot be roved. There are some parameters that do not make sense as roving parameters. For example, if we define a `go_probability` parameter that determines the probability that the next trial is a GO trial instead of a NOGO, it does not make sense to rove this value from trial-to-trial. It, however, may make sense to change this during the couse of an experiemnt (e.g., during training).

A *roving parameter* is like a parameter, except that it can be roved from trial to trial. When selected for roving, the next value of the parameter is provided by a selector.  A *selector* maintains a sequence of expressions for one or more roving parameters. In some experiments, you'll only have a single selector. In other experiments, you may want multiple selectors (e.g., one for go trials, one for remind trials and one for nogo trials). Right now, the only difference between different types of selectors will be the GUI that's presented to the user for configuring the sequence of values. Internally, all of them maintain a list of values that should be presented on successive trials.  

You can define a list of parameters required by the experiment as well as a set of selectors that allow you to specify a sequence of values to test. Once you have defined these parameters, you can iterate through them:

.. source::

    context = workbench.get_plugin('psi.context')
    context.next_setting(selector='go')
    values = context.get_values()

    context.next_setting(selector='nogo')
    values = context.get_values()

This strategy is used in the appetitive go-nogo experiment.
