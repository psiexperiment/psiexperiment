=====================
Input-output manifest
=====================

Examples
--------

Noise exposure
..............

Basic configuration of a system with one output (connectd to a speaker) and two inputs (from microphones) driven by a National Instruments DAQ card. This configuration is about as simple as it gets.

.. literalinclude:: ../../psi/templates/noise_exposure.enaml
    :language: enaml

Appetitive go-nogo behavior
...........................

Example configuration of a system designed for appetitive go-nogo behavior where the subject must nose-poke to start a trial and retrieve their reward from a food hopper. Both the nose-poke and food hopper have an infrared beam (photoemitter to photosensor) that generate an analog signal indicating the intensity of light falling on the photosensor. If the path between the photoemitter and photosensor is blocked (e.g., by the subject's nose), then the analog readout will reflect the change in light intensity.

For a go-nogo behavioral task, we need to convert this analog readout to a binary signal indicating whether the subject broke the infrared beam or not. In the following example we create a new processing chain, `AnalogToDigitalFilter` that performs this conversion and apply it to both the nose-poke and food hopper inputs.

.. literalinclude:: ../../psi/templates/gonogo_behavior.enaml
    :language: enaml
