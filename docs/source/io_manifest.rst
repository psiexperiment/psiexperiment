=====================
Input-output manifest
=====================

The Input-Output (IO) manifest is the bridge between your experimental logic and your specific hardware setup. It defines the devices (e.g., National Instruments DAQ, LabJack), engines, and channels that psiexperiment will use to generate stimuli and acquire data.

By decoupling hardware configuration into its own file, the same experimental paradigm can be run in different labs by simply swapping the IO manifest.

-----------------------
Structure of an IO File
-----------------------

An IO manifest is written in Enaml and defines an ``IOManifest`` class. It typically contains one or more **Engines** and their associated **Channels**.

.. code-block:: enaml

    from psi.controller.api import (IOManifest, NIEngine, AnalogInput, 
                                    AnalogOutput, DigitalOutput)

    enamldef MyLabIOManifest(IOManifest): manifest:
        id = 'my_lab_hardware'

        NIEngine: engine:
            name = 'NI-DAQmx'
            # Identifier of the DAQ device in NI Max
            dev_name = 'Dev1'
            master_clock = True

            AnalogOutput: speaker:
                name = 'speaker'
                channel = 'ao0'
                fs = 100000.0
                terminal_mode = 'differential'

            AnalogInput: microphone:
                name = 'microphone'
                channel = 'ai0'
                fs = 100000.0
                terminal_mode = 'differential'

            DigitalOutput: water_valve:
                name = 'water_valve'
                channel = 'port0/line0'

----------------------
Engines and Syncing
----------------------

The **Engine** is responsible for the hardware timing and control loop. You can have multiple engines in a single experiment (e.g., an NI DAQ for high-speed audio and an Arduino for simple digital triggers). 

*   **Master Clock**: One engine must be designated as the ``master_clock``. This engine provides the reference timing for the entire experimental loop.
*   **Hardware Timed**: High-speed channels (like ``AnalogInput``) are usually hardware-timed by the engine's clock.
*   **Software Timed**: Low-speed channels (like digital toggles) can be software-timed.

-----------------------
Examples from Templates
-----------------------

Psiexperiment ships with a set of templates in ``psi/templates/io/`` that demonstrate common hardware configurations.

Simple NI-DAQ Configuration
...........................

A basic configuration for a system with one analog output (connected to a speaker) and one analog input (from a microphone) driven by a National Instruments DAQ card.

.. literalinclude:: ../../psi/templates/io/_ni-bare-bones.enaml
    :language: enaml

Behavioral Setup with Digital Logic
...................................

For behavioral experiments, you may need to convert analog signals (e.g., from an infrared beam) into binary events. You can define custom processing chains within the IO manifest to handle this.

.. literalinclude:: ../../psi/templates/io/_gonogo_behavior.enaml
    :language: enaml

---------------------------
How to use an IO Manifest
---------------------------

When launching an experiment, you specify the IO manifest via the ``--io`` command-line argument:

.. code-block:: bash

    psi my_experiment --io my_lab_hardware

Psiexperiment will automatically look for the file in your configured ``IO_ROOT`` folder. For more information on setting up your environment, see :doc:`installing`.
