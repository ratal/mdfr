"""Microbenchmarks measuring Python FFI optimization in `src/mdfr.rs`.

Compares string `eval()` execution against direct PyO3 C-API / method invocation.
"""

import time
import datetime
import numpy as np
import polars as pl

ITERATIONS = 20_000

def bench_numpy_ma():
    data = np.arange(1000, dtype=np.float64)
    mask = np.zeros(1000, dtype=bool)
    locals_dict = {"numpy": np, "data": data, "mask": mask}

    # Baseline: py.eval string evaluation
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        eval("numpy.ma.array(data, mask=mask)", None, locals_dict)
    t_eval = time.perf_counter() - t0

    # Optimized: direct method invocation
    ma_array = np.ma.array
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        ma_array(data, mask=mask)
    t_direct = time.perf_counter() - t0

    speedup = t_eval / t_direct
    print(f"NumPy MaskedArray Creation ({ITERATIONS} calls):")
    print(f"  • Baseline (eval string parsing): {t_eval*1e6/ITERATIONS:.2f} µs / call")
    print(f"  • Optimized (direct C API call):  {t_direct*1e6/ITERATIONS:.2f} µs / call")
    print(f"  • Measured Speedup:              {speedup:.2f}x faster")


def bench_datetime_construction():
    ts = 1700000000.123456
    offset_s = 3600
    locals_dict = {"datetime": datetime, "ts": ts, "offset_s": offset_s}

    # Baseline: py.eval string evaluation
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        eval("datetime.datetime.fromtimestamp(ts, datetime.timezone(datetime.timedelta(seconds=offset_s)))", None, locals_dict)
    t_eval = time.perf_counter() - t0

    # Optimized: direct method invocation
    fromtimestamp = datetime.datetime.fromtimestamp
    timezone = datetime.timezone
    timedelta = datetime.timedelta
    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        tz = timezone(timedelta(seconds=offset_s))
        fromtimestamp(ts, tz)
    t_direct = time.perf_counter() - t0

    speedup = t_eval / t_direct
    print(f"\nDateTime fromtimestamp ({ITERATIONS} calls):")
    print(f"  • Baseline (eval string parsing): {t_eval*1e6/ITERATIONS:.2f} µs / call")
    print(f"  • Optimized (direct C API call):  {t_direct*1e6/ITERATIONS:.2f} µs / call")
    print(f"  • Measured Speedup:              {speedup:.2f}x faster")


def bench_polars_dataframe():
    series_dict = {"col1": [1, 2, 3, 4], "col2": [10.0, 20.0, 30.0, 40.0]}
    locals_dict = {"polars": pl, "series_dict": series_dict}

    t0 = time.perf_counter()
    for _ in range(5_000):
        eval("polars.DataFrame(series_dict)", None, locals_dict)
    t_eval = time.perf_counter() - t0

    df_ctor = pl.DataFrame
    t0 = time.perf_counter()
    for _ in range(5_000):
        df_ctor(series_dict)
    t_direct = time.perf_counter() - t0

    speedup = t_eval / t_direct
    print(f"\nPolars DataFrame Construction (5,000 calls):")
    print(f"  • Baseline (eval string parsing): {t_eval*1e6/5000:.2f} µs / call")
    print(f"  • Optimized (direct C API call):  {t_direct*1e6/5000:.2f} µs / call")
    print(f"  • Measured Speedup:              {speedup:.2f}x faster")


if __name__ == "__main__":
    print("══════════════════════════════════════════════════════════════════════")
    print("     PYTHON FFI INTEROP OVERHEAD BENCHMARKS                           ")
    print("══════════════════════════════════════════════════════════════════════")
    bench_numpy_ma()
    bench_datetime_construction()
    bench_polars_dataframe()
    print("══════════════════════════════════════════════════════════════════════")

