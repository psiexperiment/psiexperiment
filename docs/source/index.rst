=========================================
 Welcome to Psiexperiment
=========================================
Psiexperiment is a plugin-based framework for creating feature-rich auditory experiments with minimal effort. Psiexperiment can be run on any platform which supports Python and Qt. The framework is robust enough to support a diverse range of closed and open-loop experiments. So far, the following experiments have been implemented:

* Auditory brainstem responses.
* Distortion product otoacoustic emissions (input-output functions, contralateral suppression, suppression using optogenetic stimuli).
* Envelope following responses.
* Operant behavior (go-nogo task) that can embed discrete sound tokens (i.e., targets and/or distractors) in a continuous background masker.
* Noise exposures.

Screencasts
-----------
To get a better view of these screencasts, right click on the image and select "view image" in your browser.

Chirp calibration
.................

Screencast illustrating several key features of psiexperiment including:

* The dock layout (which allows individual tabs to be rearranged)
* The ability to select a parameter to acquire multiple values at (i.e., we want to acquire responses at two chirp levels, -40 and -400 dB attenuation). For calibration experiments, stimulus levels are typically specified as dB attenuation instead of dB SPL. 
* Real-time plotting and analysis of acquired data (note that the data is automatically broken down by chirp level). 

.. image:: resources/chirp_animation.gif
    :align: center
    :alt: Screencast of speaker calibration using chirp


DPOAE acquisition
.................

Another screencast showing a DPOAE acquisition experiment. As in the previous screencast, this illustrates several features including:

* Rearranging dock items (the layout can be saved for future reuse if desired).
* Selecting two parameters for which we want to acquire multiple values (i.e., F2 frequency and F2 level) and real-time plotting and analysis of acquired data (i.e., the FFT and DPOAE levels).

.. image:: resources/DPOAE_animation.gif
    :align: center
    :alt: Screencast of DPOAE acquisition


What you get
------------
:doc:`Simple hardware configuration<io_manifest>`
.................................................
* You describe the devies and data acquisition channels available for your experiment in a :doc:`configuration file<io_manifest>`. Psiexperiment will automatically configure the hardware based on the requirements of the experiment you're running. 
* Since all stimulus generation and data acquisition is built on top of a hardware abstraction layer, your experiment can easily be shared with other labs who may have a different acquisition system. Provided psiexperiment has the appropriate interface for that system and the system supports the appropriate capabilities needed by the experiment (i.e, sampling rate, number of input and output channels, etc.), psiexperiment will be able to run the experiment on that system.
* Built-in caching and buffers to optimize stimulus generation and data acquisition.

:doc:`Simple generation of auditory stimuli<waveforms>`
.......................................................
* Psiexperiment has a set of robust auditory calibration utilities using tones, chirps and Golay sequences that can be used to calibrate both closed-field and free-field systems.
* An auditory stimulus generation system that incorporates calibration information and can generate complex stimuli that are either brief (i.e., clicks and tone pips for ABRs) or near-infinite (i.e., long noise exposures, continuous background masking during behavior, etc.) in duration.
* A powerful stimulus queue that allows you to queue stimuli for playout. Any type of stimulus can be added to the queue, allowing you to build complex trial structures (e.g., trains of tone pips with varying frequency and level; tone clouds; temporal orthogonal ripple noise combinations followed by a tone).
* Support for continuous generation of stimuli that are naturally infinite. For example, noise can be infinite in duration; however, the typical approach is to generate a short segment (e.g., 30 seconds) and repeat that segment. Instead, we support the generation of infinite stimuli without having to repeat a segment.
* Support for merging multiple stimuli into a single stream that can be played through a speaker. For example, this allows generation of continuous maskers during a behavioral experiment. The target stimuli are then embedded in this masker at the appropriate times.

:doc:`Plugin-based system<creating_experiment>`
...............................................
* You can define actions based on what happens during an experiment. If the animal licks the spout, what should the program do?
* A native GUI with dockable components that reacts to user input (built on `Atom <https://atom.readthedocs.io>`_ and `Enaml <https://enaml.readthedocs.io>`_).
* Simple, intuitive experiment configuration that allows you to focus on experiment design (i.e., what you want the experiment to do) rather than implementation (how to write the code to make it work).
* :doc:`Easy configuration of experiment settings via the GUI<context>`. 

  * New settings can be added via a few lines of code
  * One or more settings can be marked for control as part of a sequence of values to be tested. For example, in many tests of peripehral auditory function multiple frequencies and levels will be tested.
  * In the GUI, values for settings can be expressed as equations. For example, if you have settings specifying the levels of two tones, `f1_level` and `f2_level`, you can fix the level of the second tone relative to the first by specifying it's value as `f2_level = f1_level + 10`. Alternatively, you could randomly draw the level from a set of values on each trial as `f2_level = np.random.choice([30, 35, 40, 45])`.

* Large number of plugins for controlling the sequence of experiments, generating various stimuli, plotting results and saving data. Plugins can modify any part of psiexperiment (e.g., each plugin can contribute one or more dockable components to the GUI, contribute one or more new stimuli types, etc.).
* Easy to write new plugins using a declarative programming langauge with Python flavour.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installing
   io_manifest
   software/index.rst

:doc:`installing`
-----------------
Instructions on installing psiexperiment and configuring your system.

:doc:`io_manifest`
------------------
Instructions on creating your input-output manifest which describes all equipment used for the experiment.

.. toctree::
   :maxdepth: 2
   :caption: Creating an experiment
   :hidden:

   waveforms
   context
   creating_experiment

:doc:`software`
---------------
Instructions for end-users.

:doc:`context`
--------------
Define parameters and sequences.

:doc:`waveforms`
----------------
Create stimuli.

:doc:`creating_experiment`
--------------------------
Instructions on how to create your own experiment.

.. toctree::
   :maxdepth: 5
   :caption: Reference
   :hidden:

   API documentation <api/modules>

:doc:`api/modules`
------------------
Overview of API.
