=========================
Waveforms and Signals
=========================

Waveform generation in psiexperiment is handled by the **Token** plugin (``psi.token``). It provides a high-performance system for generating signals ranging from simple tones and clicks to complex noise masks and chirps.

---------------------------
The Token System
---------------------------

A **Token** is a declarative block that describes a signal. Tokens are composed of **Blocks** that can be chained together to create complex waveforms. For example, you can create a "Tone Pip" by chaining a ``Tone`` block with a ``CosineEnvelope`` block.

Tokens are defined in your manifest by contributing to the ``psi.token.tokens`` extension point:

.. code-block:: enaml

    from psi.token.api import Token, Tone, CosineEnvelope

    Extension:
        id = manifest.id + '.tokens'
        point = 'psi.token.tokens'

        Token:
            name = 'tone_pip'
            # Parameters can be tied to context items
            Tone:
                freq = C.frequency
                level = C.level
            CosineEnvelope:
                duration = C.duration
                rise_time = 0.005

---------------------------
Continuous vs. Epoch Signals
---------------------------

Psiexperiment distinguishes between two primary types of signal presentation:

Continuous Signals
------------------
These are signals that have an effectively infinite duration, such as a background noise masker. 
*   **Infinite Loop**: The engine continuously pulls samples from these generators.
*   **Gapless**: Unlike many other systems, psiexperiment does not repeat short segments of noise. It generates unique samples for the entire duration of the experiment.

Epoch Signals
-------------
These are finite stimuli, such as a 5ms click or a 50ms tone pip. 
*   **On-Demand**: These signals are triggered by experimental events (e.g., ``trial_start``).
*   **Queued**: Multiple epoch stimuli can be queued for presentation at precise times relative to each other.

---------------------------
Hardware Engines and Buffers
---------------------------

The hardware engine is responsible for the actual delivery of samples to the digital-to-analog converter (DAC). 

1.  **Polling**: The engine continuously "polls" all active outputs (continuous and epoch-based) for new samples. 
2.  **Buffering**: To prevent audio glitches if the computer is momentarily busy, the engine maintains a large output buffer (typically 1-5 seconds of audio).
3.  **Zero-Fill**: If no stimulus is active (e.g., during an inter-trial interval), the engine automatically fills the buffer with zeros.

---------------------------
Controlling Waveforms
---------------------------

While signals are often triggered automatically via actions, they can also be controlled programmatically using **Commands**.

.. code-block:: python

    # Get the workbench core plugin
    core = workbench.get_plugin('enaml.workbench.core')

    # Tell an output named 'target' to prepare its next stimulus
    core.invoke_command('target.prepare')

    # Tell the output to start playing at a specific timestamp (in seconds)
    core.invoke_command('target.start', {'ts': 10.5})

By using this command-driven approach, you can synchronize stimulus presentation with other experimental events across different plugins.
