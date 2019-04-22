# Installing

## From source

To install from source (e.g., if you plan to develop):

    git clone https://github.com/bburan/psiexperiment
    pip install -e ./psiexperiment

# After installing

Create the required configuration files:

    psi-config create

To view where the configuration file was saved:

    psi-config show

Now, open that file in your preferred Python editor and update the variables to point to where you want the various files stored. By default, you can have all files created by `psiexperiment` saved under a single `BASE_DIRECTORY`. Alternatively, you may want to be more specific (e.g., log files go here, data goes there, etc.). Feel free to customize as needed.

Now, go to where you defined `IO_ROOT` and create a file that ends with the extension `.enaml`. By convention, the file name should match the hostname of your computer (e.g., if your computer is called `bobcat`, then the file would be `bobcat.enaml`); however, this is not a requirement. Inside this file, you will describe the configuration of your system using Enaml (TODO: link to some example configurations). 

