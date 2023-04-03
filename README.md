# ServerlessWorkload

## k8s.py
This file contains python code to extract the number of pods running during the experiment.

## start_test.py (MacOS, Linux) and start_test_window.py (Windows)
This is the script to start the experiment. It calls JMeter binary file (HTTP-Serverless-Request.jmx) from the command line and automatically sets the config parameters.
The script also supports generating reports automatically at the end of the execution.
The two 'servers' and 'configs' variables inside the script define the server config to run the testing each time. 
You can modify those variables depending on your testing environment.



## funcx_* files
These files are used to run the experiment for funcx serverless.


## har-cnn.py and har-lstm.py
These two files are the contents of the code used for the serverless functions.


