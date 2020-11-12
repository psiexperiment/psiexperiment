:: To activate a conda environment and run a psiexperiment command
set env=psi1
set miniconda_path=c:\Users\lbhb\miniconda3\Scripts
set cmd=psi-behavior

:: Need to use `call` to ensure we remain within the current shell
call %miniconda_path%\activate %env%

:: Now, run the command
%cmd%

:: Pause to give user a chance to review output and/or errors before exiting
pause
