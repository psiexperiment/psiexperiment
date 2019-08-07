=========
Waveforms
=========

We've already discussed this briefly. However, what we have not discussed is *how* we tell the program to present a particular waveform. The engines are *always* polling every output (continuous, epoch, queued, etc.) for samples. If an output is not active (e.g., we are in the intertrial period and therefore there is no target to present), the epoch output will simply return a string of zeros. The engine actually buffers about 10 to 30 seconds of data in the output queues (to protect the experiment from crashing when you stall the computer by checking your email).

We can activate an output called `target` to begin at 10.5 seconds (re. acquisition start):

```
core = workbench.get_plugin('enaml.workbench.core')
core.invoke_command('target.prepare')
core.invoke_command('target.start', {'ts': 10.5})
```

Commands are basically strings that map to curried functions (one of Ivar's favorite types of functions).
