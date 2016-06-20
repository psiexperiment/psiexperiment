ψ<sub>experiment</sub>
======================

Psiexperiment is a plugin-based experiment controller that facilitates the
process of writing experiments. There are four core plugins: context,
experiment, controller and data.

Terminology
-----------

### Context

Every experiment has a set of variables that define the behavior. These
variables range from the stimulus frequency and level to the intertrial
interval duration. Sometimes these variables need to be expressed as functions
of other variables, or the value of the variable needs to vary in a random
fashion.

A *context item* provides information about a value that is managed by the
context plugin. When defining a context item in one of your plugin manifests,
you will provide basic information about the item (e.g., a GUI label, compact
GUI label and numpy dtype). This information will be used by plugins that
interact with the context plugin (for example, the name and dtype of the
context item will be used by the HDF5Store plugin to set up the table that
stores acquired trial data).

There are currently three specific types of context items. A *result* is a
value provided by a plugin. It cannot be defined by the user. Common use cases
may include the values computed after a trial is acquired (e.g., one can
compute the `reaction_time` and provide it as a result).

A *parameter* is a value that can be configured by the user before and during
an experiment. While the value of the parameter can be modified by the user
during the experiment, it cannot be roved. There are some parameters that do
not make sense as roving parameters. For example, if we define a
`go_probability` parameter that determines the probability that the next trial
is a GO trial instead of a NOGO, it does not make sense to rove this value from
trial-to-trial. It, however, may make sense to change this during the couse of
an experiemnt (e.g., during training).

A *roving parameter* is like a parameter, except that it can be roved from
trial to trial. When selected for roving, the next value of the parameter is
provided by a selector.

A *selector* maintains a sequence of expressions for one or more roving
parameters. In some experiments, you'll only have a single selector. In other
experiments, you may want multiple selectors (e.g., one for go trials, one for
remind trials and one for nogo trials). Right now, the only difference between
different types of selectors will be the GUI that's presented to the user for
configuring the sequence of values. Internally, all of them maintain a list of
values that should be presented on successive trials.

### Controller

The controller is the central plugin of the experiment system. It's responsible
for managing a series of *engines* and *actions*. 

An *engine* communicates with a data acquisition device (e.g., a NI-DAQmx
card). Each engine manages a set of analog and digital *channels*. The channel
declaration contains sufficient information (e.g., expected range of values,
sampling frequency, start trigger for acquisition, etc.) for the engine to
configure the channel. Each channel has a set of *inputs* or *outputs*
associated with it (depending on whether it's an input or output channel). The
inputs and outputs act as the primary interface to the hardware for various
psiexpermient plugins. 


Plugins
-------

### Context plugin

Manages the context items and selectors (see terminology above).  Every
experiment has a set of values, or parameters, that define the course of the
experiment.  This plugin provides a central registry of all context items,
manages their current values (i.e., when notified, the current values will be
cleared and re-computed for the next trial).  A subset of these context items
can be specified as roving parameters (i.e., their value varies from trial to
trial in a predefined way). Values for roving parameters are managed by one or
more selectors.  Selectors are objects that provide a mechanism for specifying
the sequence of values that the context plugin draws from on each trial. 

### Experiment plugin

Provides basic management of the experiment workspace by managing GUI elements
provided by the loaded plugins (e.g., the context plugin provides several GUI
widgets) as well as providing methods to save/restore the layout of the
workspace as well as save/restore the preferences for each plugin.

### Data plugin

Provides basic management and analysis of data acquired during an experiment.

### Controller plugin

TODO


Roadmap
-------

TODO

Migrating from Neurobehavior 
----------------------------

This is a significant rewrite of Neurobehavior that leverages the strengths of
the Atom/Enaml framework. The key change is that Psiexperiment is plugin-based.
Writing experiments in Neurobehavior involved subclassing several classes
(e.g., paradigm, experiment, controller, data) and incorporating various mixins
(e.g., for the pump controller and pump data). Often this required some
cumbersome hacks to get the GUI and experiment to behave the way you want.
There were also some significant limitations with the context management system
(i.e., adding new parameters required subclassing the paradigm class and
possibly incorporating mixins for pumps, etc.). 

In contrast, setting up new experiments in psiexperiment should be simpler. You
decide the types of plugins you want loaded and write a manifest file that
defines the specific extensions you want. If you have a specific type of
analysis that you need done on the acquired data, you can write a new plugin
and load it.
