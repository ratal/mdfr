# mdfr-derive

Procedural macro crate providing `#[derive(MdfRecord)]` for **mdfr** and **mdfr-core**.

It analyzes standard Rust `struct` definitions at compile time, computes their little-endian memory layout, generates channel metadata (name, unit, comment, linear/rational conversion formulas), and implements the `mdfr_core::logger::MdfRecord` trait with zero runtime reflection overhead.

---

## Usage

Add `mdfr-core` with the default `derive` feature enabled (or use `mdfr` on desktop):

```toml
[dependencies]
mdfr-core = { version = "0.8", default-features = false, features = ["derive"] }
```

### Deriving a Record

```rust
use mdfr_core::MdfRecord;

#[derive(MdfRecord)]
#[mdf(group = "Powertrain")]
pub struct VehicleStatus {
    // Linear conversion: physical = 0.25 * raw + 0.0
    #[mdf(name = "EngineSpeed", unit = "rpm", linear(0.0, 0.25))]
    pub rpm: u16,

    // Rational conversion: physical = (p1*X^2 + p2*X + p3) / (p4*X^2 + p5*X + p6)
    #[mdf(name = "Torque", unit = "Nm", rational(0.0, 0.5, -100.0, 0.0, 0.0, 1.0))]
    pub torque: i16,

    // Unconverted channels inherit their Rust field name and type
    #[mdf(unit = "%")]
    pub throttle: f32,

    pub gear: u8,
    pub fault_active: bool,

    // Fixed-size raw byte arrays
    pub raw_can_frame: [u8; 8],
}
```

### Registering and Logging

```rust
use mdfr_core::logger::{DataLayout, CompressionAlgorithm, Logger, LoggerBuilder};
use mdfr_core::time::ManualClock;

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let mut builder = LoggerBuilder::new(DataLayout::Sorted, CompressionAlgorithm::NoCompression);

    // Register all channels and conversion blocks automatically
    let group = builder.add_record_group::<VehicleStatus>()?;
    let schema = builder.build()?;

    let mut logger = Logger::start(schema, std::fs::File::create("log.mf4")?, ManualClock::new(0.001, 1))?;

    let sample = VehicleStatus {
        rpm: 3200,
        torque: 400,
        throttle: 45.0,
        gear: 4,
        fault_active: false,
        raw_can_frame: [0x12, 0x34, 0x56, 0x78, 0x00, 0x00, 0x00, 0x00],
    };

    // Serialize and push the record with 1 function call
    logger.push_record(group, &sample)?;

    logger.close()?;
    Ok(())
}
```

---

## Supported Attributes

### Struct Attributes
- `#[mdf(group = "GroupName")]`: Overrides the default channel group name (which otherwise defaults to the struct name).

### Field Attributes
- `#[mdf(name = "ChannelName")]`: Custom channel name in the MDF file (defaults to field name).
- `#[mdf(unit = "UnitString")]`: Measurement unit (e.g., `"m/s"`, `"rpm"`, `"degC"`).
- `#[mdf(comment = "Description")]`: Human-readable channel description stored in the MDF `TXBLOCK`.
- `#[mdf(linear(offset, slope))]`: Linear physical value conversion (`p1 = offset`, `p2 = slope`).
- `#[mdf(rational(p1, p2, p3, p4, p5, p6))]`: Rational physical value conversion formula:
  $$\text{phys} = \frac{p_1 X^2 + p_2 X + p_3}{p_4 X^2 + p_5 X + p_6}$$

---

## Supported Field Types

- Unsigned integers: `u8`, `u16`, `u32`, `u64`
- Signed integers: `i8`, `i16`, `i32`, `i64`
- Floating-point: `f32`, `f64`
- Boolean: `bool` (stored as 1-byte LE unsigned int)
- Byte arrays: `[u8; N]` (stored as `CnDataType::ByteArray`)
