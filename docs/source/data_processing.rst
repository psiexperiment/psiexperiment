=========================
Data Processing Pipeline
=========================

Psiexperiment uses a sophisticated pipeline system for handling incoming and outgoing data. This ensures that signal processing is efficient, traceable, and modular.

---------------------------
Input Processing
---------------------------

All analog input data is streamed through a series of operations that perform transformations in real-time. This is defined in your **IO Manifest**.

Data Objects
------------
Data is passed through the pipeline as specialized NumPy-like objects that carry additional metadata, such as the sampling rate and timestamp of the first sample.

Pipeline Clearing
-----------------
If an engine needs to reset (e.g., after a hardware error or an experiment restart), a special "clear" signal is sent through the pipeline to ensure that any buffered filters or accumulators are reset.

---------------------------
Calibration
---------------------------

Psiexperiment was built with auditory research in mind, so calibration is a first-class citizen. Every input and output channel can have an associated calibration object.

How Calibration Works
---------------------
When you specify a stimulus level (e.g., 80 dB SPL), the system uses the calibration data to calculate the required voltage for the hardware output. Conversely, when acquiring data from a microphone, it uses the calibration to convert measured voltages back into sound pressure levels.

Special Calibration Classes
---------------------------

**UnityCalibration**
Used when you want to express or measure values relative to 1V (dB re 1V). This is useful for non-acoustic signals or when a system is already pre-calibrated.

Example usage:
*   ``calibration.get_sf(frequency, level_db)``: Calculates the scale factor (V) for a given frequency and level.
*   ``calibration.get_spl(frequency, voltage)``: Calculates the level (dB) for a given frequency and voltage.

---------------------------
Data Sinks and Storage
---------------------------

Data is saved by contributing a **Sink** to the ``psi.data.sinks`` extension point.

*   **Continuous Sinks**: Save every sample acquired (e.g., to a Zarr or HDF5 file).
*   **Event Sinks**: Save discrete experimental events and their associated context values (e.g., to a CSV or SQLite database).
*   **Trial Sinks**: Automatically extract epochs around specific triggers and save them as independent trials.

By combining these sinks, you can ensure that you have both the raw high-speed data and a summarized version of the experiment for quick analysis.
