//! Static library entry point for C firmware integration.
//!
//! Produces `libmdfr_core_c.a` (or `libmdfr_core.a`) for linking into bare-metal
//! C toolchains (e.g. `arm-none-eabi-gcc`).
#![no_std]
#![no_main]

// Re-export all C API symbols from mdfr-core so they are included in the static archive.
pub use mdfr_core::capi::*;

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
