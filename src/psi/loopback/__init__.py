'''
Tests that run paradigms on a sound card whose outputs are routed back to its
inputs (e.g., an RME Fireface with loopbacks set up in TotalMix), so what a
paradigm says it played can be checked against what it recorded.

A project describes its sound card with a `rig.Rig` (which IO manifest the
paradigms run on and which channel is which) and pulls in the pytest plugin
from the conftest of its hardware tests::

    # tests/hardware/conftest.py
    import pytest
    from psi.loopback.pytest_plugin import *  # noqa: F403
    from psi.loopback.rig import Rig

    @pytest.fixture(scope='session')
    def rig():
        return Rig(io_manifest='...', device='ASIO Fireface USB',
                   outputs={...}, inputs={...})

The tests are then skipped unless pytest is run with `--fireface`. Each run
writes its data, psi's log and a report under a results folder (see
`pytest_plugin`).

Modules:

* `rig`: the sound card, and a probe that measures which output reaches
  which input, at what gain and delay.
* `runner`: runs a paradigm unattended in a subprocess (`autorun`).
* `fit`: fits the stimuli a paradigm logged to what it recorded.
* `continuity`: a pilot signal for finding dropped or repeated buffers, and
  `pilot.PilotManifest`, which plays and records it during a run.
* `report`, `plots`: the HTML report for each run.
'''
