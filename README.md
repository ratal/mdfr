# mdfr
ASAM mdf reader in rust

Currently a personal project to learn rust from past experience with python (mdfreader).

mdfr is currently able to be used from python interpreter (using pyO3) to read mdf 3.x and 4.x files. Packaging should come later.
Using rayon crate on many parts of the code allows to have faster parsing in a safe and easy way compared to python.
It can be used the following way:

```python
import mdfr
obj = mdfr.Mdf('path/to/file.mdf')
# loads all data into memory, data could be accessed on demand by other methods
obj.load_all_channels_data_in_memory()
# Returns the data, saves it also in obj
obj.get_channel_data('channel_name')
# Methods to retrieve channel description, unit, related master name, type and data,
# list of channels in file, or lists of channels grouped by master are available
```

mdfr on the other hand does not include much features to manipulate files comparitively to mdfreader. You can for now only write a mdf4.x to mdf4.2 (relatively slow for now, should be parallelised). Convertion from mdf3 to mdf4 is in the todo list along with resampling, cutting data and exporting to hdf5 and parquet.
