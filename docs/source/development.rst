====================
Development Workflow
====================

Helpful resources
.................

Psiexperiment leverages `Enaml`_ both for building the user-interface and for implementing a plugin-based system. Extending psiexperiment requires familiarity with the `Enaml Workbench plugin framework`_.

Setting up your environment
...........................

The recommended way to develop psiexperiment is to use an editable install within a dedicated virtual environment. 

1. Clone the repository
-----------------------

.. code-block:: bash

    git clone https://github.com/bburan/psiexperiment
    cd psiexperiment

2. Create a virtual environment
-------------------------------

Using ``venv``:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Using ``conda``:

.. code-block:: bash

    conda create -n psi-dev python=3.10
    conda activate psi-dev

3. Install in editable mode
---------------------------

Install psiexperiment along with its development, documentation, and testing dependencies:

.. code-block:: bash

    pip install -e .[dev,docs,test]

4. Configure your environment
-----------------------------

Set the ``PSI_CONFIG_FILE`` environment variable to point to a local configuration file for development. This prevents your development work from interfering with any production installs, and it is also how each conda environment gets its own configuration: point the variable at a different file before launching.

.. code-block:: bash

    # On Windows (PowerShell)
    $env:PSI_CONFIG_FILE = "C:/path/to/your/dev/config.toml"

    # On Linux/macOS
    export PSI_CONFIG_FILE="/path/to/your/dev/config.toml"

Then, initialize your development folders:

.. code-block:: bash

    psi-config create --base-directory ./dev_root
    psi-config create-folders

Running Tests
.............

Tests are written using ``pytest``. To run the full test suite:

.. code-block:: bash

    pytest tests

Testing paradigms through a loopback
------------------------------------

``psi.loopback`` runs paradigms unattended on a sound card whose outputs are
routed back to its inputs (e.g., an RME Fireface set up in TotalMix), so what
a paradigm says it played can be checked against what it recorded. cfts and
abts use it for their hardware tests (``tests/hardware`` in each). It
provides:

* ``rig``: describes the sound card and measures which output reaches which
  input, at what gain and delay.
* ``runner``: runs a paradigm in a subprocess with the test's settings, and
  any extra manifests the test needs, then reports how it ended.
* ``fit``: fits the stimuli a paradigm logged to the recording.
* ``continuity`` and ``pilot.PilotManifest``: plays a known noise on a spare
  output for the whole run and finds any buffer that was dropped, repeated
  or zeroed between the sound card and psi's stores (the sound card's own
  under/overrun flags can't be relied on). The pilot also records how far
  psi's input processing falls behind the sound card.
* ``report``: an HTML page of checks and figures for each run.
* ``pytest_plugin``: the ``--fireface`` option (the tests are skipped
  without it) and the session fixtures. See its docstring for how to use it
  from a project's ``conftest.py``.

Building Documentation
......................

Documentation is built using Sphinx. To generate the HTML version:

.. code-block:: bash

    cd docs
    make html

The output will be located in ``docs/build/html``.

Updating API Documentation
--------------------------

To regenerate the API reference files from the source code:

.. code-block:: bash

    cd docs
    sphinx-apidoc ../psi -o source/api

Contributing
............

1. **Branching**: Create a new feature branch for your changes.
2. **Coding Standards**: Follow PEP 8 and ensure all new Enaml code follows the declarative patterns established in the core plugins.
3. **Tests**: Include new tests for any features or bug fixes.
4. **Pull Requests**: Submit a PR to the main repository for review.

.. _Enaml: http://enaml.readthedocs.io/en/latest/
.. _Enaml Workbench Plugin Framework: https://enaml.readthedocs.io/en/latest/dev_guides/workbenches.html
