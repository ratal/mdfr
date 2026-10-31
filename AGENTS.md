# mdfr Workspace Guidelines & Architectural Invariants

This document outlines key architectural boundaries, build patterns, and development guidelines for the `mdfr` project.

---

## 1. Workspace Crate Architecture & Boundaries

The project is structured as a Cargo workspace with distinct separation of concerns:

- **`mdfr-core`**:
  - Strictly `#![no_std]` by default (opt-in `alloc` and `std` features).
  - Must **NEVER** depend on PyO3 or Python C-API dependencies.
  - Implements the MDF 4.3 streaming logger (`MdfLogger`), low-level block structures (`HeaderBlock`, `IdBlock`, `ChannelBlock`, etc.), time bases (`TimeSource`, `StdClock`), compression codecs (Deflate, LZ4), and IO adapters (`Sink`, `EmbeddedIoSink`).
  - Target compatibility: Host OS (std) and bare-metal/RTOS embedded targets (e.g., ARM Cortex-M4F `thumbv7em-none-eabihf`).

- **`mdfr-derive`**:
  - Procedural macro crate implementing `#[derive(MdfRecord)]`.
  - Kept lightweight, operating on struct field attributes (`#[mdf(name = "...", channel_type = "...")]`) to generate `MdfRecord` trait implementations for `mdfr-core`.

- **`mdfr`**:
  - High-level crate with full standard library (`std`).
  - Contains PyO3 Python bindings, Polars dataframe conversions, parallel file readers, and Python wrappers (`MdfLogger`, `Mdfr`).
  - Re-exports core types where appropriate and implements conversions (e.g., `From<PyCompressionAlgorithm> for CompressionAlgorithm`).

---

## 2. Standalone C Static Library (`staticlib`) Pattern

When building a C static library (`libmdfr_core_c.a`) for embedded C/C++ integration:

- **Do NOT** add `crate-type = ["staticlib"]` directly to `[lib]` in `mdfr-core/Cargo.toml`.
  - Doing so forces `rustc` to require a `#[global_allocator]` and `#[panic_handler]` on **every** build of `mdfr-core` (including pure `rlib` downstream consumers and `cargo check --no-default-features`).
- **Pattern**: Use an example target with `crate-type = ["staticlib"]`:
  ```toml
  [[example]]
  name = "mdfr_core_c"
  crate-type = ["staticlib"]
  path = "examples/c_staticlib.rs"
  ```
- In `c_staticlib.rs`, gate the bare-metal panic handler behind `#[cfg(not(feature = "std"))]` to avoid duplicate lang item conflicts when compiling or testing with `--all-features`.

---

## 3. Codebase Consistency & Single Source of Truth

To prevent divergence and maintainability debt:

- **MDF Block Definitions**:
  - All MDF 4.3 block structures, headers, block flags, and binary serializers belong exclusively in `mdfr-core/src/blocks/`.
  - Higher-level parsing, Polars schema extraction, and block traversals reside in `mdfr/src/mdfinfo/mdfinfo4/*_impl.rs`.
- **Shared Algorithms & Utilities**:
  - Avoid duplicate logic across crates. For example, DZ block column transposition / untransposition (`untranspose_after_decompression`) and clocks (`StdClock`) belong in `mdfr-core` behind appropriate feature flags (`alloc`, `std`).
  - Block unit tests belong in `mdfr-core/src/blocks/` alongside block definitions.

---

## 4. Python Support & Packaging

- **Supported Python Versions**:
  - Python 3.9 is End-of-Life (EOL) and removed. Support is strictly Python `>= 3.10`.
  - All CI workflows (`py-release-*.yml`) and `pyproject.toml` must enforce `>= 3.10`.
- **Typing & Stubs**:
  - Maintain `mdfr.pyi` and `py.typed` at repository root. Any new public Python classes, methods, or enums must be reflected in `mdfr.pyi`.
- **Crates.io Metadata**:
  - Workspace crates (`mdfr`, `mdfr-core`, `mdfr-derive`) share version synchronization.
  - Keep `publish = false` until official crates.io release readiness.

---

## 5. Verification & Quality Assurance

Before committing changes, pushing, or submitting pull requests:

1. **Formatting**: Must pass formatting checks before commits:
   - `cargo fmt --check` (or format in place via `cargo fmt`)
2. **Clippy**: Must pass with 0 warnings across all feature combinations:
   - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - `cargo clippy -p mdfr-core --no-default-features -- -D warnings`
3. **Embedded no_std verification**:
   - `cargo check -p mdfr-core --target thumbv7em-none-eabihf --no-default-features --features alloc`
   - `cargo build -p mdfr-core --example mdfr_core_c --target thumbv7em-none-eabihf --no-default-features --features alloc`
4. **Tests**:
   - `cargo test --workspace --all-features`
