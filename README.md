# mdfr
ASAM mdf reader in rust

Currently a personal project to learn rust from past experience with python (mdfreader).

mdfr is currently able to be used from python interpreter (using pyO3) to read mdf 3.x and 4.x files. You can install it with 'pip install mdfr' command.
Using rayon crate on many parts of the code allows to have faster parsing in a safe and easy way compared to python.
To allow efficient data sharing with many other tools, mdfr stores the data using arrow2. Polars (pandas equivalent) use is therefore also straight forward.
It can be used the following way with python interpreter:

```python
import mdfr
# load file metadata in memory
obj = mdfr.Mdfr('path/to/file.mdf')
# loads all data in memory
obj.load_all_channels_data_in_memory()
# loads a set of channels in memory, for instance in case there is not enough free memory or for performance
obj.load_channels_data_in_memory({'Channel1', 'Channel2'})
# Returns the numpy array
obj.get_channel_data('channel_name')

# .get_channel_* methods to retrieve channel description, unit, related master name, type and data
# .set_channel_* methods to modify channel description, unit, related master name, type and data
# list of channels in file, or lists of channels grouped by master are available
obj.get_master_channel_names_set()

# To manipulate the file data (cut, resample, merge, etc.), it is possible to use polars:
obj.get_polars_series('channel_name') # returns the a serie
# to get complete dataframe including given channel:
obj.get_polars_dataframe('channel_name')
# add and remove channel
obj.add_channel(channel_name, data, master_channel, master_type, master_flag, unit, description)
# to plot a single channel with matplotlib:
obj.plot('channel_name')

# Export to parquet:
obj.export_to_parquet('file_name', compression_option)
# write to mdf4 file, compressed or not
obj.write('file_name', conpression_flag)
```

A C/C++ api is also available allowing to get list of channels, units, description and data using the Arrow CDataInterface.
