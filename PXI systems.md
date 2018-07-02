Installing
----------

1. Install [Miniconda](https://conda.io/miniconda.html). Be sure to choose Python 3.6 or greater. Use the default (recommended) settings proposed by the installer.

2. An Anaconda prompt should now appear on the desktop and/or in the start menu. Open this prompt.

3. Create an environment called "psi" (to keep it separate from testing versions that I may give you).  The `-c intel` tells the package manager to install Intel-optimized versions of the numerical libraries. Omit this flag if you are not on an Intel platform.
   ```
	conda create -n psi -c ecpy -c intel python=3 numpy scipy pandas atom enaml pyyaml joblib pyqtgraph bcolz
	```

4. Activate the environment.
	```
	activate psi
	```

5. Install additional packages using pip (these are not availble via the conda
   repos).
   ```
	pip install pydaqmx palettable
	```

6. Clone the git repository for [psiexperiment](https://github.com/bburan/psiexperiment). If you prefer not to install [Git](https://git-scm.com/download/win) on your system, you can download a (zip archive)[https://github.com/bburan/psiexperiment/archive/master.zip] and extract it to a folder. If you decided to install Git, see

7. Now, CD to the folder where you cloned the git repository and install psiexperiment. You need to be in the parent folder containing the repository folder (e.g., if you saved the repository to `c:/users/lab/python/psi/psiexperiment`, you need to CD to `c:/users/lab/python/psi`. Note the `-e` flag!
	```
	pip install -e psiexperiment
	```

8. Create a file called `psiexperiment/psi/application/io/<hostname>.enaml` and update it with your hardware configuration (usually you can just copy `pika.enaml` and update the AI and AO channel information to point to the correct PXI cards). Eventually I plan to update the code so that you can create a configuration file that lives outside the main repository.

9. Create an environment variable called `PSIEXPERIMENT_BASE` pointing to the folder where you want all data to be saved (e.g., `d:/psi`)

10. Find where the `cfts.exe` program is saved (usually in `c:\users\lab\Miniconda3\envs\<name of env>\Scripts`) and make a shortcut to it on the desktop. This can also be launched via the Anaconda prompt, but only after activating the psi environment, e.g.:
	```
	activate psi
	cfts
	```

11. To run the program, double-click on the shortcut.
