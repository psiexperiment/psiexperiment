:: To copy two files from local path to remote path 
robocopy "<local path>" "<remote path>" "<local file>" "<local file>" /XO

:: To mirror all files in local path to remote path. This does not delete any
:: files in the remote path (allowing you to clean up the local folder as
:: needed).
robocopy "<local path>" "<remote path>" /NP /IPG:5 /XO /R:5 /FFT /E
