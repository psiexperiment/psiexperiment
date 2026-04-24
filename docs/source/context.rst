=======
Context
=======

The **Context** plugin (``psi.context``) is one of the most powerful features of psiexperiment. It manages every variable that defines experimental behavior—from stimulus frequency and level to intertrial interval durations and randomization rules.

By using the Context system, you ensure that your experimental parameters are:
1.  **Observable**: The GUI and other plugins can react when a value changes.
2.  **Persistent**: Settings are automatically saved and loaded between sessions.
3.  **Traceable**: The value of every parameter is recorded alongside your experimental data.

---------------------
Types of Context Items
---------------------

A *context item* is a single variable managed by the system. There are three primary types:

Parameters
----------
A **Parameter** is a value that can be configured by the user via the GUI. 
*   **Static Parameters**: These maintain a constant value throughout the experiment (though the user can change them manually).
*   **Editable**: Parameters can be updated while the experiment is running.

Roving Parameters
-----------------
A **Roving Parameter** is like a standard parameter, but it can be controlled by a **Selector** to change from trial to trial. 
*   **Sequences**: When a parameter is "roved," its next value is drawn from a sequence (e.g., a list of frequencies or randomized levels).
*   **Selectors**: A selector manages the sequence for one or more roving parameters. You might have one selector for "Target" trials and another for "Distractor" trials.

Results
-------
A **Result** is a value provided by a plugin *after* an event occurs (e.g., a calculated reaction time or a peak amplitude). 
*   **Read-Only**: Results cannot be edited by the user in the GUI.
*   **Data Storage**: Like parameters, results are automatically captured by data sinks.

--------------------------
Expressions and Math
--------------------------

One of the unique features of the context system is that parameter values can be expressed as **mathematical formulas**.

*   **Variables**: You can define one parameter in terms of another. For example, if you have ``f1_level``, you can set ``f2_level = f1_level + 10``.
*   **Functions**: You can use built-in functions like ``db()`` or standard NumPy/SciPy functions (e.g., ``np.random.choice([30, 40, 50])``).
*   **Dynamic**: Formulas are re-evaluated whenever the dependent variables change.

--------------------------
Defining Context Items
--------------------------

Context items are defined in your manifest by contributing to the ``psi.context.items`` extension point.

.. code-block:: enaml

    from psi.context.api import Parameter, Result

    Extension:
        id = manifest.id + '.context_items'
        point = 'psi.context.items'

        Parameter:
            name = 'intertrial_interval'
            label = 'ITI (s)'
            default = 2.0
            group_name = 'timing'

        Result:
            name = 'trial_score'
            label = 'Score'

----------------------------
Iterating through Settings
----------------------------

In your experiment controller, you interact with the context plugin to move to the next set of values in a sequence.

.. code-block:: python

    # Get the context plugin
    context = workbench.get_plugin('psi.context')

    # Advance the 'default' selector to the next trial setting
    context.next_setting(selector='default')

    # Retrieve all current parameter values as a dictionary
    values = context.get_values()
    print(f"Current frequency: {values['frequency']}")

By advancing the selector, the context plugin updates all roving parameters to their next values and notifies the rest of the system (including hardware engines and the GUI).
