Input processing pipeline
-------------------------
All analog input data is streamed through a series of operations that perform
transformations on the data.

Handling input data
-------------------
All data is passed as an `InputData` object, which is a subclass of Numpy's
ndarray. `InputData` has an extra attribute, `metadata` which contains
information about the data segment.

Alternatively, you may receive a special flag, `Ellipsis`. This flag indicates
that the pipeline must be cleared.
