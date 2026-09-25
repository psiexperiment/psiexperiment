=============
Configuration
=============

A setting has exactly one spelling. The name used in code is the name in
``config.toml`` and the name of the environment variable, package prefix
included::

    get_config('PSI_DATA_ROOT')     # in code
    PSI_DATA_ROOT = "D:/bulk"       # in config.toml
    set PSI_DATA_ROOT=D:/bulk       # in the environment

How a value is resolved
=======================

Three layers, last one wins:

1. **The default built into the package.** Every setting has one, so
   psiexperiment and the applications built on it run with no
   configuration file at all.
2. **The configuration file**, ``config.toml``.
3. **The environment.**

The environment beating the file is the ordinary arrangement — pip, git,
conda and the AWS CLI all resolve settings this way — and it means a
one-off override never requires editing a file that is shared or
version-controlled.

To see where a value actually came from::

    psi-config show

which prints every setting, its resolved value, and the layer that
supplied it. That listing is the first thing to check when a setting is
not what you expect.

Where the configuration file lives
==================================

``PSI_CONFIG_FILE`` names the file. If it is not set, the file is
``~/psi/config.toml``.

Because it selects the file, ``PSI_CONFIG_FILE`` is the one variable that
cannot be set inside the file. It is the only bootstrap variable: there is
no separate "configuration folder" setting, and no directory is ever
inferred from the configuration file's location. Every directory is its
own named setting.

Per-environment configuration
-----------------------------

To keep separate settings per conda environment, point
``PSI_CONFIG_FILE`` at a different file before launching — typically from
the batch file or shortcut that activates the environment and starts the
application. There is no layering and no auto-detection: one file is
selected, and that file is the configuration.

Editing settings
================

By hand, in any text editor, or from the command line::

    psi-config set PSI_DATA_ROOT D:/bulk-storage

Comments and formatting in a hand-edited file survive a programmatic
write, and the file is replaced atomically, so an interrupted write cannot
leave a truncated configuration behind.

If a setting is currently being forced by an environment variable,
``psi-config set`` still writes the file but warns that the write will not
take effect until the variable is cleared. Applications with a settings
GUI do the equivalent: cftscal disables a control whose setting the
environment is overriding, and names the variable responsible.

Settings
========

Paths
-----

Most paths derive from ``PSI_BASE_DIRECTORY``, so setting that one moves
the rest with it. Set an individual root only when it belongs somewhere
else entirely — bulk data on a second drive, for instance.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Setting
     - Default
     - Contents
   * - ``PSI_BASE_DIRECTORY``
     - ``~/Documents/psi``
     - Root of every other psiexperiment path.
   * - ``PSI_DATA_ROOT``
     - ``<base>/data``
     - Experiment data.
   * - ``PSI_LOG_ROOT``
     - ``<base>/logs``
     - Log files.
   * - ``PSI_PROCESSED_ROOT``
     - ``<base>/processed``
     - Post-processed experiment data.
   * - ``PSI_PREFERENCES_ROOT``
     - ``<base>/preferences``
     - Experiment-specific saved settings.
   * - ``PSI_LAYOUT_ROOT``
     - ``<base>/layout``
     - Experiment-specific window layouts.
   * - ``PSI_IO_ROOT``
     - ``<base>/io``
     - IO manifests describing attached hardware.

Experiment discovery
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Setting
     - Default
     - Contents
   * - ``PSI_PARADIGM_DESCRIPTIONS``
     - the descriptions shipped with psi
     - Modules containing paradigm descriptions. An experiment is only
       discoverable when its module is listed, so an installation that
       adds paradigms must list them here.
   * - ``PSI_STANDARD_IO``
     - ``[]``
     - Hardware configurations offered in addition to whatever is found
       in ``PSI_IO_ROOT``.
   * - ``PSI_HOSTNAME``
     - this machine's hostname
     - Used to select a hostname-specific IO manifest.

Hardware
--------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Setting
     - Default
     - Contents
   * - ``PSI_NI_EEG_CHANNEL``
     - ``PXI1Slot8/ai0``
     - Read by the shipped PXIe-1062 IO template.
   * - ``PSI_NI_CALIBRATION_CHANNEL``
     - ``PXI1Slot7/ai0``
     - As above.
   * - ``PSI_NI_STARSHIP_CHANNEL``
     - ``PXI1Slot7/ai1``
     - As above.
   * - ``PSI_NI_START_TRIGGER``
     - ``/PXI1Slot7/ao/StartTrigger``
     - As above.
   * - ``PSI_WEBSOCKETS_URI``
     - ``''``
     - Endpoint for the websocket paradigm mixins.

Settings owned by other packages
--------------------------------

Packages built on psiexperiment register their own settings under their
own prefix, and they resolve identically. ``CFTS_ROOT`` (cfts) is where
the cfts launcher keeps its saved presets; ``CFTSCAL_ROOT`` (cftscal) is
where calibration data is stored. See each package's documentation.

What is *not* a setting
=======================

Two categories of name look like settings and are not.

**Runtime values.** The experiment name, the log filename, the parsed
command-line arguments and the profiling flag are decided by the launcher
after the process starts. They live in :mod:`psi.runtime`, have no
configuration-file or environment spelling, and setting something like
``PSI_LOG_FILENAME`` does nothing.

**Handoff variables.** ``CFTSCAL_*`` variables other than ``CFTSCAL_ROOT``
and the ``PSI_SOUND_DEVICE_*`` pair are a process contract: an application
writes them into the environment of the ``psi`` subprocess it launches,
and the paradigm manifests read them there. They are not configuration,
they do not belong in ``config.toml``, and they last only as long as the
subprocess. cftscal documents that contract.

Type conversion
===============

Environment variables are always strings, and TOML has no path type, so a
value is converted to the type of the setting's default. ``Path``,
``int`` and ``float`` convert as you would expect. Booleans accept
``1``/``0``, ``true``/``false``, ``yes``/``no`` and ``on``/``off``; a
value that is none of those is an error rather than a silent ``True``.
Lists are native arrays in TOML and comma-separated in the environment.

Adding a setting
================

Settings are declared in one table per package — ``psi/config_defaults.py``
and its equivalents — and registered with
:func:`psi.config.register_defaults`. Two rules matter:

* **Every setting needs a usable default**, because a missing
  configuration file is a supported state.
* **Defaults are zero-argument callables, not values.** A default derived
  from another setting (all the psi roots are derived from
  ``PSI_BASE_DIRECTORY``) has to resolve *after* the configuration file is
  read. Resolving at import would freeze the built-in base directory into
  the derived values, so setting ``PSI_BASE_DIRECTORY`` in the file would
  silently fail to move anything else.

Do not pass a ``default=`` argument to ``get_config`` for a setting that
has a registered default. That is how one setting ends up with two
different defaults in two modules; a test enforces it.

Migrating from ``config.py``
============================

Earlier versions used an executable ``config.py`` whose keys had no
package prefix and which, unintuitively, took precedence over the
environment. There is no backwards compatibility: a legacy name is simply
not read.

Before upgrading a machine, find the legacy names on it::

    python tools/audit_legacy_config.py

The auditor is stand-alone — standard library only, no ``psi`` import — so
it runs on a machine whose environment is half-upgraded, and it parses
configuration files without executing them. It exits non-zero when it
finds anything, so it can gate a deployment script.

Then convert::

    psi-config migrate path/to/config.py
    psi-config show

``migrate`` executes the old file once to capture computed values, maps
the names, and folds in cftscal's ``workspace.json`` and per-plugin
calibration settings if they are present.
