//! Time source trait for the embedded logger.
//!
//! The logger owns a [`TimeSource`] and stamps every record on push. The master time
//! channel stores raw integer ticks (`u32` or `u64`) paired with a linear `CCBLOCK`
//! converting ticks to seconds (`phys = P1 + P2 * raw`), so the MCU never needs floating-point
//! math on the hot path.

/// Clock abstraction owned by the logger.
pub trait TimeSource {
    /// Monotonic tick counter at the moment a record is pushed.
    fn now_ticks(&mut self) -> u64;

    /// Seconds per tick (stored as `P2` in the master channel's linear conversion block).
    /// E.g. `1e-6` for a 1 MHz microsecond timer, `1e-3` for a 1 kHz millisecond tick.
    fn seconds_per_tick(&self) -> f64;

    /// Optional absolute UTC start timestamp in nanoseconds since `1970-01-01T00:00:00Z`.
    /// Written to `hd_start_time_ns` at `start()` or patched at `checkpoint()`/`close()` once
    /// RTC/GNSS lock becomes available.
    fn wall_clock_ns(&mut self) -> Option<u64> {
        None
    }
}

/// Simple deterministic counter-based [`TimeSource`] (useful for tests, fixed-rate logging, or manual ticks).
#[derive(Debug, Clone)]
pub struct ManualClock {
    /// Current tick value.
    pub ticks: u64,
    /// Auto-increment applied on each `now_ticks()` call.
    pub step_ticks: u64,
    /// Seconds per tick (`P2` in linear CC).
    pub sec_per_tick: f64,
    /// Optional wall-clock start time in ns since UNIX epoch.
    pub start_time_ns: Option<u64>,
}

impl ManualClock {
    /// Creates a manual clock with `sec_per_tick` resolution and `step_ticks` increment per record.
    pub const fn new(sec_per_tick: f64, step_ticks: u64) -> Self {
        Self {
            ticks: 0,
            step_ticks,
            sec_per_tick,
            start_time_ns: Some(1_700_000_000_000_000_000),
        }
    }
}

impl TimeSource for ManualClock {
    fn now_ticks(&mut self) -> u64 {
        let t = self.ticks;
        self.ticks = self.ticks.wrapping_add(self.step_ticks);
        t
    }

    fn seconds_per_tick(&self) -> f64 {
        self.sec_per_tick
    }

    fn wall_clock_ns(&mut self) -> Option<u64> {
        self.start_time_ns
    }
}
