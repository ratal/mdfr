# mdfr-core

`mdfr-core` is the core, `no_std + alloc` engine of **mdfr**. It provides exact ASAM MDF 4.3 binary block models, standard codecs (Deflate, LZ4, byte transposers), a high-performance streaming logger designed for embedded microcontrollers and RTOS environments (ARM Cortex-M4/M7/A), a zero-copy static arena allocator, and a stable C FFI.

---

## Features

- **`no_std + alloc` by Default**: Zero desktop dependencies (`arrow`, `rayon`, `polars`, `pyo3` are strictly avoided). Compatible with bare-metal targets (`thumbv7em-none-eabihf`, etc.).
- **Complete MDF 4.3 Block Model**:
  - Strongly typed definitions for `ID`, `HD`, `FH`, `DG`, `CG`, `CN`, `CC`, `CA`, `SI`, `CH`, `EV`, `SR`, `AT`, `TX`, `DT`, `DL`, `DZ`, `HL`, `LD`, `GD`.
  - Strict 24-byte header parsing, 8-byte link alignment, and little-endian binary serialization via `binrw`.
- **High-Performance Streaming Logger**:
  - **Data Layouts**:
    - `DataLayout::Sorted`: Independent `DTBLOCK` per group, or block-streamed `DZBLOCK` / `DLBLOCK` with transposition.
    - `DataLayout::Unsorted`: Single interleaved record stream with 1/2/4/8-byte record IDs, automatic DL-block chunking, and companion `CG_F_VLSC` CGBLOCKs for variable-length signals.
  - **Compression**: `NoCompression`, `Lz4Transpose`, and `DeflateTranspose`.
  - **Fast Push Engines**:
    - Typed builder: `logger.begin_record(group)?.set_u64(ch, val)?.commit()?`.
    - Hardware/bus frame logging: `logger.push_raw_at(group, ticks, &bytes)`.
    - Struct logging: compile-time record layout with `#[derive(MdfRecord)]` via `mdfr-derive`.
  - **Lock-Free ISR Logging**: Built-in [`SpscRing`](src/logger/spsc.rs) single-producer single-consumer ring buffer for interrupt-safe logging.
  - **Bus Presets**: Pre-configured schemas for CAN (`can`), CAN-FD (`can_fd`), LIN (`lin`), FlexRay (`flexray`), and Ethernet (`ethernet`).
  - **Power-Loss Recovery**: In-place finalization and mid-record truncation repair with `recover_in_place(&mut sink)`.
- **Bare-Metal Memory Management**:
  - Optional zero-fragmentation [`StaticArena`](src/arena.rs) (`features = ["capi-arena"]`) with free-list coalescing for systems without a heap.
- **Embedded Observability**:
  - Optional `defmt` logging integration (`features = ["defmt"]`).
- **C FFI**:
  - Stable C function-pointer vtable interface (`include/mdfr_logger.h`) for FatFs, SD card drivers, POSIX, or bare-metal UART.

---

## Cargo Features

| Feature | Description | Default |
| :--- | :--- | :--- |
| `deflate` | Miniz-oxide Deflate compression (`no_std + alloc`) | **Yes** |
| `lz4` | LZ4-flex block compression (`no_std + alloc`) | **Yes** |
| `derive` | Re-exports `#[derive(MdfRecord)]` from `mdfr-derive` | **Yes** |
| `std` | Links with `std` (`thiserror/std`, `binrw/std`, `chrono/std`) | No |
| `zstd` | Zstandard block compression (requires `std`) | No |
| `capi` | Compiles the `extern "C"` FFI API | No |
| `capi-arena` | Includes `StaticArena` bare-metal memory allocator | No |
| `defmt` | Implements `defmt::Format` for error types | No |

---

## Rust Usage Example

```rust
use mdfr_core::logger::{
    CompressionAlgorithm, ConversionDef, DataLayout, Logger, LoggerBuilder,
};
use mdfr_core::blocks::cn::CnDataType;
use mdfr_core::time::ManualClock;
use mdfr_core::MdfRecord;

// 1. Derive schema layout for compile-time packed records
#[derive(MdfRecord)]
#[mdf(group = "Engine")]
struct EngineFrame {
    #[mdf(name = "RPM", unit = "rpm", linear(0.0, 0.25))]
    rpm_raw: u16,
    #[mdf(name = "CoolantTemp", unit = "degC", linear(-40.0, 1.0))]
    temp_raw: u8,
    #[mdf(name = "WarningActive")]
    warning: bool,
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    // 2. Configure schema
    let mut builder = LoggerBuilder::new(
        DataLayout::Sorted,
        CompressionAlgorithm::Lz4Transpose,
    );
    let g_eng = builder.add_record_group::<EngineFrame>()?;

    // Also supports dynamic channels, bit-fields, arrays, and VLSC:
    let g_diag = builder.add_group("Diagnostics");
    let ch_text = builder.add_vlsc_channel(g_diag, "FaultLog", CnDataType::StringUtf8)?;

    let schema = builder.build()?;

    // 3. Start logging to any sink implementing mdfr_core::io::Sink (e.g. std::fs::File, Vec<u8>)
    let output = std::fs::File::create("measurement.mf4")?;
    let clock = ManualClock::new(0.001, 1); // 1 ms resolution
    let mut logger = Logger::start(schema, output, clock)?;

    // 4. Stream data at runtime
    let frame = EngineFrame {
        rpm_raw: 4000, // physical 1000 rpm
        temp_raw: 125, // physical 85 degC
        warning: false,
    };
    logger.push_record(g_eng, &frame)?;

    logger
        .begin_record(g_diag)?
        .set_bytes(ch_text, b"No error codes detected")?
        .commit()?;

    // 5. Finalize file header cleanly
    logger.close()?;
    Ok(())
}
```

---

## On-Device Power-Loss Recovery

If power is cut abruptly mid-recording without calling `close()`, the file on flash or SD card can be finalized and truncated in-place on the embedded target without requiring desktop tools:

```rust
let mut file = std::fs::OpenOptions::new().read(true).write(true).open("measurement.mf4")?;
let recovered_records = mdfr_core::logger::recover_in_place(&mut file)?;
println!("Repaired and finalized {recovered_records} records in-place.");
```

---

## C API (`include/mdfr_logger.h`)

`mdfr-core` exports a clean, non-panicking C interface. Bind any underlying storage via `mdfr_sink_t` and clock via `mdfr_clock_t`:

```c
#include "mdfr_logger.h"

// Define I/O vtable (e.g. using FatFs)
static int my_write(void* ctx, const uint8_t* buf, size_t len) {
    UINT bw;
    return f_write((FIL*)ctx, buf, len, &bw) == FR_OK ? 0 : -1;
}

static int my_seek(void* ctx, uint64_t pos) {
    return f_lseek((FIL*)ctx, pos) == FR_OK ? 0 : -1;
}

// Build schema & stream
mdfr_builder_t* b = mdfr_logger_builder_new(MDFR_LAYOUT_SORTED, MDFR_COMPRESS_LZ4_TRANSPOSE);
uint16_t group = 0;
mdfr_logger_add_group(b, "Engine", &group);
uint32_t ch_rpm = 0;
mdfr_logger_add_channel(b, group, "RPM", MDFR_CN_DATA_TYPE_UINT_LE, 16, &ch_rpm);

mdfr_sink_t sink = { .ctx = &fil, .write = my_write, .seek = my_seek, .flush = NULL, .close = NULL };
mdfr_clock_t clock = { .ctx = NULL, .now_ticks = get_timer_ticks, .tick_ns = 1000000, .wall_clock_ns = NULL };

mdfr_logger_t* logger = NULL;
mdfr_logger_start(b, sink, clock, &logger);

// Record sample
mdfr_logger_record_begin(logger, group);
mdfr_logger_record_set_u64(logger, ch_rpm, 3200);
mdfr_logger_record_commit(logger, group);

mdfr_logger_close(logger);
```
