# mdfr

High-performance ASAM MDF (Measurement Data Format) 3.x and 4.x reader, writer, and embedded streaming logger in Rust.

`mdfr` provides fast, memory-efficient access to automotive measurement data across desktop analytics, server pipelines, and resource-constrained embedded targets.

---

## Key Capabilities

- **High-Performance Reader**:
  - Full support for **ASAM MDF 3.x and 4.x** (`.mdf`, `.mf4`).
  - Multithreaded parallel parsing with **Rayon** and zero-copy data representation using **Apache Arrow**.
  - Direct integration with **Polars** DataFrames and **NumPy** arrays.
  - Exporters for **Parquet**, **HDF5**, and **CSV**.
- **Embedded MDF 4.3 Streaming Logger (`no_std + alloc`)**:
  - Independent `mdfr-core` subcrate engineered for microcontrollers and RTOS environments (ARM Cortex-M4/M7/A).
  - High-throughput streaming with both `Sorted` (independent DT/DZ blocks) and `Unsorted` (single interleaved stream) layouts.
  - Zero-copy hardware bus presets for **CAN**, **CAN-FD**, **LIN**, **FlexRay**, and **Ethernet**.
  - Lock-free ISR interrupt logging via single-producer single-consumer ring buffer ([`SpscRing`](mdfr-core/src/logger/spsc.rs)).
  - Power-loss resilience: mid-record truncation repair and on-device in-place file recovery ([`recover_in_place`](mdfr-core/src/logger/recover.rs)).
- **Compile-Time Struct Logging**:
  - `#[derive(MdfRecord)]` procedural macro (`mdfr-derive`) to serialize custom Rust structs with automatic physical conversion and unit metadata.
- **MDF 4.3 Variable-Length Channels (VLSC)**:
  - First-class support for `cn_type = 7` (Variable Length with companion Size Channel) across sorted `VDBLOCK` and unsorted `CG_F_VLSC` streams.
- **Multilingual Support**:
  - Python bindings via PyO3 (`pip install mdfr`).
  - Stable C FFI for desktop reading (`mdfr.h`) and embedded logging (`include/mdfr_logger.h`).

---

## Workspace Architecture

| Crate | Target / Environment | Description |
| :--- | :--- | :--- |
| **[`mdfr`](.)** | Desktop / Server (`std`) | Full-featured crate with Arrow, Polars, Rayon, Parquet, HDF5, Python bindings, and CLI. |
| **[`mdfr-core`](mdfr-core)** | Embedded / Bare-metal (`no_std + alloc`) | Pure MDF 4.3 block model, Deflate/LZ4 codecs, static arena allocator, and streaming logger engine. |
| **[`mdfr-derive`](mdfr-derive)** | Proc-Macro | `#[derive(MdfRecord)]` for compile-time record layout and zero-overhead serialization. |

---

## Python Usage

Install from PyPI:
```bash
pip install mdfr
```

### Reading Files and Polars Integration
```python
import mdfr

# Load MDF 3 or 4 file metadata
obj = mdfr.Mdfr("measurement.mf4")

# Load all channels into memory (Arrow-backed)
obj.load_all_channels_data_in_memory()

# Or load a selective subset for low memory footprint
obj.load_channels_data_in_memory({"EngineSpeed", "VehicleSpeed"})

# Retrieve numpy array
speed = obj.get_channel_data("VehicleSpeed")

# Retrieve Polars Series or complete DataFrame grouped by master channel
series = obj.get_polars_series("VehicleSpeed")
df = obj.get_polars_dataframe("VehicleSpeed")

# Export to other analytical formats
obj.export_to_parquet("output.parquet", compression="snappy")
obj.export_to_hdf5("output.h5")
obj.export_to_csv("output.csv")
```

### Streaming Logger (Python)
```python
import mdfr

# Configure a streaming logger
builder = mdfr.MdfLogger(layout="sorted", compression="lz4")
g_vehicle = builder.add_group("Vehicle")
ch_speed = builder.add_channel(g_vehicle, "Speed", "float", 64)
builder.set_channel_unit(ch_speed, "km/h")

logger = builder.start("streamed_log.mf4")

for i in range(100):
    logger.begin_record(g_vehicle)
    logger.set_f64(ch_speed, 50.0 + i * 0.5)
    logger.commit()

logger.close()
```

---

## Rust Usage

### 1. Analytical Desktop Reader (`mdfr` with Polars)

```toml
[dependencies]
mdfr = { version = "0.8", features = ["polars", "parquet"] }
```

```rust
use mdfr::mdfreader::Mdf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mdf = Mdf::new("measurement.mf4")?;
    mdf.load_all_channels_data_in_memory()?;

    // Extract Polars DataFrame grouped by master channel
    let df = mdf.get_channel_polars_dataframe(Some("Time"))?;
    println!("{df}");

    // Export to Parquet
    mdf.export_to_parquet("measurement.parquet", mdfr::export::parquet::CompressionOption::Snappy)?;

    Ok(())
}
```

### 2. Embedded Streaming Logger (`mdfr-core` + `mdfr-derive`)

```toml
[dependencies]
mdfr-core = { version = "0.8", default-features = false, features = ["lz4", "derive"] }
```

```rust
use mdfr_core::logger::{CompressionAlgorithm, DataLayout, Logger, LoggerBuilder};
use mdfr_core::time::ManualClock;
use mdfr_core::MdfRecord;

#[derive(MdfRecord)]
#[mdf(group = "Powertrain")]
struct PowertrainFrame {
    #[mdf(name = "RPM", unit = "rpm", linear(0.0, 0.25))]
    rpm_raw: u16,
    #[mdf(name = "Torque", unit = "Nm", rational(0.0, 0.5, -100.0, 0.0, 0.0, 1.0))]
    torque_raw: i16,
    #[mdf(name = "Throttle", unit = "%")]
    throttle: f32,
    #[mdf(name = "FaultActive")]
    fault: bool,
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::Lz4Transpose);
    let g_pt = builder.add_record_group::<PowertrainFrame>()?;
    let schema = builder.build()?;

    let file = std::fs::File::create("log.mf4")?;
    let clock = ManualClock::new(0.001, 1); // 1 ms resolution
    let mut logger = Logger::start(schema, file, clock)?;

    let sample = PowertrainFrame {
        rpm_raw: 3200,   // phys: 800 rpm
        torque_raw: 400, // phys: 100 Nm
        throttle: 25.0,
        fault: false,
    };

    // Zero-overhead push
    logger.push_record(g_pt, &sample)?;
    logger.close()?;
    Ok(())
}
```

### 3. In-Place Power-Loss Recovery

In embedded applications, unexpected power cut before `logger.close()` leaves the MDF 4.3 header unfinalized. `recover_in_place` repairs the file directly on the storage device:

```rust
let mut file = std::fs::OpenOptions::new().read(true).write(true).open("log.mf4")?;
let recovered_samples = mdfr_core::logger::recover_in_place(&mut file)?;
println!("Repaired {recovered_samples} records in-place without data loss.");
```

---

## C / C++ Embedded ABI (`include/mdfr_logger.h`)

`mdfr-core` exposes a clean, non-panicking C API using function-pointer vtables for I/O (`mdfr_sink_t`) and timer clocks (`mdfr_clock_t`), suitable for FatFs, LittleFS, POSIX, or bare-metal ring buffers:

```c
#include "mdfr_logger.h"

// Define storage vtable (e.g. FatFs f_write / f_lseek)
mdfr_sink_t sink = { .ctx = &fil, .write = my_write, .seek = my_seek, .flush = NULL, .close = NULL };
mdfr_clock_t clock = { .ctx = NULL, .now_ticks = get_ticks, .tick_ns = 1000000, .wall_clock_ns = NULL };

mdfr_builder_t* b = mdfr_logger_builder_new(MDFR_LAYOUT_SORTED, MDFR_COMPRESS_LZ4_TRANSPOSE);
uint16_t group = 0;
mdfr_logger_add_group(b, "CAN_Raw", &group);

mdfr_logger_t* logger = NULL;
mdfr_logger_start(b, sink, clock, &logger);

// Push raw hardware-timestamped bus frame
uint8_t payload[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
mdfr_logger_push_raw_at(logger, group, get_hardware_timer(), payload, sizeof(payload));

mdfr_logger_close(logger);
```

---

## Cargo Features

### `mdfr` (Desktop)
- `default = ["numpy", "parquet", "polars", "cli"]`
- `polars`: High-performance Polars DataFrame integration.
- `parquet`: Apache Parquet exporter.
- `hdf5`: HDF5 exporter (`rust-hdf5`).
- `numpy`: NumPy array conversion via PyO3.
- `cli`: Command-line executable (`mdfr`).

### `mdfr-core` (Embedded)
- `default = ["deflate", "lz4", "derive"]`
- `deflate`: Pure Rust Miniz-Oxide Deflate compression (`no_std + alloc`).
- `lz4`: Pure Rust LZ4 compression (`no_std + alloc`).
- `derive`: Proc-macro `#[derive(MdfRecord)]` integration.
- `capi-arena`: Zero-fragmentation `StaticArena` bare-metal memory allocator.
- `defmt`: Implements `defmt::Format` for embedded logging.

---

## License

GPL-3.0 License.
