====================
Development Workflow
====================

Helpful resources
.................

Psiexperiment leverages `Enaml`_ both for building the user-interface as well as implementing a plugin-based system. Extending psiexperiment requires familiarity with the `Enaml Workbench plugin framework`_.

Setting up your environment
...........................

The following instructions assume that you use a conda-based distribution (e.g., Anaconda or Miniconda). This enables you to manage multiple versions of psiexperiment (e.g., a "production" version and a "development" version). 

First, create an environment containing the dependencies required by psiexperiment. Substitute your preferred environment name (e.g., `psi-dev`) for `ENV`.

.. code-block:: doscon

    conda create -n ENV -c psiexperiment psiexperiment --only-deps

Now, install the psiexperiment source code. The `pip install` command explicitly installs dependencies for building documentation (e.g., `sphinx`) and testing (e.g., `pytest`):

.. code-block:: doscon

    mkdir %USERPROFILE%\projects\psi-dev\src
    cd %USERPROFILE\projects\psi-dev\src
    git clone https://github.com/bburan/psiexperiment
    pip install -e ./psiexperiment[docs,test]

Create environment-specific environment variables and folders for your source code as needed. You can customize to your preferred workflow, but I find this particular workflow works well:

.. code-block:: doscon

    mkdir %USERPROFILE%\projects\psi-dev
    conda env config vars set -n psi-dev PSI_CONFIG=%USERPROFILE%/projects/psi-dev/conf

Follow the instructions in the command prompt to reactivate your environment and load the environment variable you just set, i.e.,:

.. code-block:: doscon

    conda activate psi-dev

Make sure it worked properly:

.. code-block:: doscon

    psi-config show

Now create your config as described in the :doc:`install instructions<installing>`.

Building documentation
......................

Documentation is built using Sphinx.

.. code-block:: doscon

    cd %USERPROFILE%\projects\psi-dev\src\psiexperiment\docs
    make html

To update the API documentation:

.. code-block:: doscon

    cd %USERPROFILE%\projects\psi-dev\src\psiexperiment\docs
    sphinx-apidoc ../psi -o source/api

It also automatically gets built on ReadTheDocs whenever new commits are pushed to the main branch on Github.

.. _Enaml: http://enaml.readthedocs.io/en/latest/
.. _Enaml Workbench Plugin Framework: https://enaml.readthedocs.io/en/latest/dev_guides/workbenches.html
