# Input and output pipelines 

## Calibration 

### Special calibration classes 

#### UnityCalibration 

This is useful when you want to express, or measure, values as dB re 1V, e.g.,:

    >>> calibration = UnityCalibration() 
    >>> calibration.get_sf(1000, 0)
    1
    >>> calibration.get_sf(1000, 20)
    10
    >>> calibration.get_sf(1000, 40)
    100

    >>> calibration.get_spl(1000, 1)
    0
    >>> calibration.get_spl(1000, 10)
    20
    >>> calibration.get_spl(1000, 100)
    40

This does highlight that the `get_spl` method is a bit of a misnomer. However, psiexperiment was designed from the ground-up as asn auditory acquisition framework so many methods are named along those lines.


## Input processing pipeline

All analog input data is streamed through a series of operations that perform
transformations on the data.

### Handling input data

All data is passed as an `InputData` object, which is a subclass of Numpy's
ndarray. `InputData` has an extra attribute, `metadata` which contains
information about the data segment.

Alternatively, you may receive a special flag, `Ellipsis`. This flag indicates
that the pipeline must be cleared.
