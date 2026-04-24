========================
Welcome to Psiexperiment
========================

Psiexperiment is a plugin-based framework for creating feature-rich auditory experiments with minimal effort. Built on top of the `Enaml Workbench <https://enaml.readthedocs.io>`_ and `Atom <https://atom.readthedocs.io>`_ libraries, it provides a robust, declarative architecture for designing both closed-loop and open-loop experiments.

The framework is designed to be highly modular, allowing researchers to share and reuse components across different hardware setups and experimental paradigms.

What you get
============

:doc:`A powerful, flexible user interface<screenshots>`
-------------------------------------------------------
Psiexperiment provides a modern, dockable interface. Panes can be rearranged, "torn off" into separate windows, or resized to fit your specific workflow. This flexibility ensures that you can monitor exactly what you need during an experiment.

Hardware Abstraction Layer
--------------------------
* **Unified Configuration**: Hardware is described in a :doc:`configuration file (IO manifest)<io_manifest>`. This decouples your experiment logic from the specific hardware being used.
* **Portability**: Experiments can be shared across labs with different acquisition systems (e.g., National Instruments, TDT, LabJack) without rewriting the core experimental logic.
* **Optimized Performance**: Built-in caching and asynchronous buffering ensure that stimulus generation and data acquisition remain high-performance even under heavy processing loads.

Advanced Stimulus Generation
----------------------------
* **Calibration Support**: Built-in utilities for acoustic calibration using tones, chirps, and Golay sequences.
* **Complex Trial Structures**: A powerful stimulus queue supports building sophisticated trials, such as randomized tone clouds or interleaved stimulus pips.
* **Infinite Waveforms**: Support for truly infinite stimulus generation (e.g., background noise) without repeating short segments.
* **Signal Merging**: Easily combine multiple independent signal streams into a single output channel.

Extensible Plugin System
------------------------
* **Declarative Logic**: Use a Python-flavored declarative language (Enaml) to define how your experiment behaves.
* **Reactive UI**: The interface automatically updates in response to changes in experiment state or user input.
* **Hooks and Extensions**: Every part of the system is extensible. You can add new plots, data storage backends, UI components, or hardware drivers by writing small, focused plugins.
* **Flexible Parameters**: :doc:`Configure experiment settings via the GUI<context>`. Parameters can be expressed as mathematical equations or randomized using Python expressions.

Getting started
===============

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

:doc:`software/index`
---------------------
Instructions for end-users on how to launch and manage experiments.
Development
===========

.. toctree::
   :maxdepth: 2
   :caption: Creating an experiment
   :hidden:

   architecture
   plugin_reference
   development
   paradigm_overview
   creating_experiment
   data_processing
   waveforms
   context

.. toctree::
   :maxdepth: 5
   :caption: Reference
   :hidden:

   API documentation <api/psi>
   API modules <api/modules>

:doc:`architecture`
-------------------
Deep dive into the plugin-based architecture and experimental lifecycle.

:doc:`plugin_reference`
-----------------------
A comprehensive guide to all core plugins and their extension points.

:doc:`development`
------------------
How to set up a development environment for contributing to psiexperiment.

:doc:`paradigm_overview`
------------------------
Understanding how experiments (paradigms) are structured.

:doc:`creating_experiment`
--------------------------
Step-by-step guide to building your first experiment from scratch.

:doc:`data_processing`
----------------------
How data is streamed, processed, and saved.

:doc:`waveforms`
----------------
Creating and presenting waveforms.

:doc:`context`
--------------
Configuring experiment settings and trial sequences.

:doc:`api/modules`
------------------
Detailed reference for the core Python API.

Contributors and Acknowledgements
=================================

* Brad Buran (New York University, Oregon Health & Science University)
* Decibel Therapeutics, Inc.

Work on psiexperiment was supported by grants R01-DC009237 and R21-DC016969 from the `National Institute on Deafness and Other Communication Disorders <https://www.nidcd.nih.gov/>`_ and an Emerging Research Grant from the `Hearing Health Foundation <https://hearinghealthfoundation.org/>`_.
