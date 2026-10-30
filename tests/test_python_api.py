"""End-to-end Python integration tests for `mdfr.MdfLogger` and `mdfr.Mdfr`."""

import tempfile
import os
import mdfr


def test_python_logger_and_reader_roundtrip():
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


if __name__ == "__main__":
    test_python_logger_and_reader_roundtrip()
    print("All python API tests passed!")
