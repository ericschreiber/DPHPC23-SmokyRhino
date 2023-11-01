# Benchmark Pipeline

# Execution
All the scripts and their description is in the Pipeline folder. 

# Add timers to your function
To only time the execution of the sddmm and not the things we do around it you can start the timer when you think its suiting. Start a run with
```
this->start_run();
``` 
and end it with
```
this->stop_run();
```
Please only use the correct timer function (eg. cpu or gpu)