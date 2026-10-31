"""End-to-end Python integration tests for `mdfr.MdfLogger` and `mdfr.Mdfr`."""

import tempfile
import os
import unittest
import mdfr


class TestPythonApi(unittest.TestCase):
    def test_python_logger_and_reader_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".mf4", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".mf4", delete=False) as tmp2:
            p1 = tmp1.name
            p2 = tmp2.name

        try:
            # 1. Create and write with MdfLogger
            logger = mdfr.MdfLogger(
                p1,
                layout="sorted",
                compression=mdfr.CompressionAlgorithm.NoCompression,
            )

            g0 = logger.add_group("Engine")
            ch_rpm = logger.add_channel(g0, "RPM", data_type=0, bit_count=16, unit="rpm")
            ch_temp = logger.add_channel(g0, "Temp", data_type=4, bit_count=32, unit="degC")
            ch_msg = logger.add_vlsd_channel(g0, "DiagMsg", data_type=7)
            ch_vlsc = logger.add_vlsc_channel(g0, "VlscPayload", data_type=7)

            logger.start()

            for i in range(25):
                logger.record_begin(g0)
                logger.record_set_u64(ch_rpm, 1000 + i * 100)
                logger.record_set_f64(ch_temp, 80.0 + i * 0.5)
                logger.record_set_bytes(ch_msg, f"diag_{i}".encode("utf-8"))
                logger.record_set_bytes(ch_vlsc, f"vlsc_{i}".encode("utf-8"))
                logger.record_commit(g0)

            # Rotate to p2
            logger.rotate(p2)
            logger.close()

            # 2. Read back p1 with Mdfr
            reader = mdfr.Mdfr(p1)
            assert reader.get_version() >= 400

            names = reader.get_channel_names_set()
            assert "time" in names
            assert "RPM" in names
            assert "Temp" in names
            assert "DiagMsg" in names

            assert reader.get_channel_unit("RPM") == "rpm"
            assert reader.get_channel_unit("Temp") == "degC"
            assert reader.get_channel_master("RPM") == "time"

            reader.load_all_channels_data_in_memory()
            rpm_data = reader.get_channel_data("RPM")
            assert len(rpm_data) == 25
            assert rpm_data[0] == 1000
            assert rpm_data[24] == 3400

            # Polars DataFrame conversion
            df = reader.get_polars_dataframe("RPM")
            assert df is not None
            assert "RPM" in df.columns
            assert "time" in df.columns
            assert len(df) == 25

            # Parquet export
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as pq_tmp:
                pq_path = pq_tmp.name
            try:
                reader.export_to_parquet(pq_path, compression="snappy")
                assert os.path.getsize(pq_path) > 0
            finally:
                if os.path.exists(pq_path):
                    os.remove(pq_path)

            # Finalize file in place on p2
            recovered = mdfr.finalize_file_in_place(p2)
            assert recovered >= 0

        finally:
            if os.path.exists(p1):
                os.remove(p1)
            if os.path.exists(p2):
                os.remove(p2)

    def test_polars_write_and_arrow_capsule_interop(self):
        import polars as pl
        import pyarrow as pa
        import numpy as np

        df = pl.DataFrame({
            "time": [0.0, 0.5, 1.0, 1.5, 2.0],
            "Speed": [10.5, 20.0, 35.5, 40.0, 55.2],
            "Gear": [1, 1, 2, 2, 3],
            "Driver": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
        })

        with tempfile.NamedTemporaryFile(suffix=".mf4", delete=False) as tmp:
            out_path = tmp.name

        try:
            # 1. Test mdfr.write_polars
            written_mdf = mdfr.write_polars(
                df,
                out_path,
                time_col="time",
                compression=mdfr.CompressionAlgorithm.Deflate,
            )
            assert written_mdf is not None

            # 2. Context manager & dictionary methods
            with mdfr.Mdfr(out_path) as m:
                assert len(m) >= 4
                assert "time" in m
                assert "Speed" in m
                assert "Gear" in m
                assert "Driver" in m
                assert "NonExistent" not in m

                keys = m.keys()
                assert "time" in keys
                assert "Speed" in keys
                assert list(iter(m)) == keys

                # PyChannel indexing
                speed_ch = m["Speed"]
                assert isinstance(speed_ch, mdfr.PyChannel)
                assert speed_ch.name == "Speed"
                assert len(speed_ch) == 5
                assert speed_ch[0] == 10.5
                assert speed_ch[-1] == 55.2
                assert "<PyChannel 'Speed'" in repr(speed_ch)

                # NumPy conversion
                np_arr = speed_ch.to_numpy()
                assert isinstance(np_arr, np.ndarray)
                assert np.allclose(np_arr, [10.5, 20.0, 35.5, 40.0, 55.2])
                np_arr2 = np.asarray(speed_ch)
                assert np.allclose(np_arr2, np_arr)

                # Arrow PyCapsule zero-copy on single channel
                pa_arr = pa.array(speed_ch)
                assert isinstance(pa_arr, pa.Array)
                assert pa_arr.to_pylist() == [10.5, 20.0, 35.5, 40.0, 55.2]

                # Polars Series conversion via PyCapsule
                pl_s = pl.Series("Speed", speed_ch)
                assert isinstance(pl_s, pl.Series)
                assert pl_s.to_list() == [10.5, 20.0, 35.5, 40.0, 55.2]
                pl_s2 = speed_ch.to_polars()
                assert pl_s2.to_list() == [10.5, 20.0, 35.5, 40.0, 55.2]

                # Arrow PyCapsule on whole Mdfr (StructArray & Stream)
                pa_tbl = pa.table(m)
                assert isinstance(pa_tbl, pa.Table)
                assert "time" in pa_tbl.column_names
                assert "Speed" in pa_tbl.column_names
                assert pa_tbl.num_rows == 5

                # Polars DataFrame from whole Mdfr via Arrow PyCapsule
                pl_df = pl.DataFrame(m)
                assert isinstance(pl_df, pl.DataFrame)
                assert "time" in pl_df.columns
                assert "Speed" in pl_df.columns
                assert "Driver" in pl_df.columns
                assert pl_df["Driver"].to_list() == ["Alice", "Bob", "Charlie", "Dave", "Eve"]

                # polars from_arrow (DataFrame in Polars 1.x or struct Series in Polars 2.x)
                pl_from_arrow = pl.from_arrow(m)
                assert isinstance(pl_from_arrow, (pl.DataFrame, pl.Series))
                if isinstance(pl_from_arrow, pl.DataFrame):
                    assert "Speed" in pl_from_arrow.columns
                    assert len(pl_from_arrow) == 5
                else:
                    assert len(pl_from_arrow) == 5
                    assert "Speed" in pl_from_arrow.struct.fields

                # polars from_arrow from PyArrow Table (constructed via capsule)
                pl_df_from_tbl = pl.from_arrow(pa_tbl)
                assert isinstance(pl_df_from_tbl, pl.DataFrame)
                assert "Speed" in pl_df_from_tbl.columns
                assert len(pl_df_from_tbl) == 5

                # items and values
                items = m.items()
                assert len(items) == len(keys)
                assert all(isinstance(v, mdfr.PyChannel) for _, v in items)

        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
