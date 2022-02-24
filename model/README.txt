Basically the only difference between any of the <method>_train.py files is in the way the
data is loaded and fed into the model

~~~~ FILES ~~~~
chunk_train.py
    - trains by loading a chunk of the full training data into memory, training, then discarding
      and loading the next chunk.

lazy_train.py
    - trains by lazy loading all the data. This is good for anything besides Colab, which throttles
      file read and write operations after a certain threshold.

memory_train.py 
    - trains by loading as much data as specified into memory then only training on that.