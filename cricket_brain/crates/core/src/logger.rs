// SPDX-License-Identifier: AGPL-3.0-only
use core::fmt;

/// Minimal logging abstraction that works in `no_std` environments.
pub trait CricketLogger {
    /// Log a formatted message.
    fn log_fmt(&mut self, args: fmt::Arguments<'_>);

    /// Log a simple event message.
    fn log_event(&mut self, event: &str) {
        self.log_fmt(format_args!("{event}"));
    }

    /// Log spike amplitude information.
    fn log_spike(&mut self, amplitude: f32) {
        self.log_fmt(format_args!("spike:{amplitude:.3}"));
    }
}

/// Logger implementation that discards all messages.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullLogger;

impl CricketLogger for NullLogger {
    fn log_fmt(&mut self, _args: fmt::Arguments<'_>) {}
}

/// Read-only telemetry hooks for observing simulation behavior.
///
/// Implementations should avoid mutating simulation state and use these hooks
/// only for monitoring/export.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TelemetryEvent {
    Spike {
        neuron_id: usize,
        timestamp: u64,
    },
    ResonanceChange {
        neuron_id: usize,
        value: f32,
    },
    SequenceMatched {
        pattern_id: usize,
        confidence: f32,
        snr: f32,
        jitter: f32,
        tolerance: f32,
    },
    SnrReport {
        ratio: f32,
    },
    SystemOverload {
        entropy: f32,
        active_neurons: usize,
        total_neurons: usize,
    },
}

/// Telemetry hooks for observing simulation behavior without mutating state.
///
/// Implement this trait to receive structured events from `CricketBrain` or
/// `SequencePredictor` during processing. The default method implementations
/// delegate to [`Telemetry::on_event`], so overriding that single method is
/// sufficient to capture all events.
pub trait Telemetry {
    /// Structured telemetry event hook.
    fn on_event(&mut self, _event: TelemetryEvent) {}

    /// Called when a neuron emits a spike at `timestamp`.
    fn on_spike(&mut self, neuron_id: usize, timestamp: u64) {
        self.on_event(TelemetryEvent::Spike {
            neuron_id,
            timestamp,
        });
    }

    /// Called when a neuron's resonance level changes.
    fn on_resonance_change(&mut self, neuron_id: usize, value: f32) {
        self.on_event(TelemetryEvent::ResonanceChange { neuron_id, value });
    }

    /// Called when a sequence/pattern match has been detected.
    fn on_sequence_match(&mut self, pattern_id: usize) {
        self.on_sequence_matched(pattern_id, 0.0, 0.0, 0.0, 1.0);
    }

    /// Called when a sequence/pattern match has been detected with certainty metrics.
    fn on_sequence_matched(
        &mut self,
        pattern_id: usize,
        confidence: f32,
        snr: f32,
        jitter: f32,
        tolerance: f32,
    ) {
        self.on_event(TelemetryEvent::SequenceMatched {
            pattern_id,
            confidence,
            snr,
            jitter,
            tolerance,
        });
    }

    /// Called when a signal-to-noise ratio estimate is available.
    fn on_snr_report(&mut self, ratio: f32) {
        self.on_event(TelemetryEvent::SnrReport { ratio });
    }

    /// Called when the system is likely in an overload/noise-chaos state.
    fn on_system_overload(&mut self, entropy: f32, active_neurons: usize, total_neurons: usize) {
        self.on_event(TelemetryEvent::SystemOverload {
            entropy,
            active_neurons,
            total_neurons,
        });
    }
}

/// Telemetry sink that ignores all events.
///
/// In debug builds, it tracks spike timestamps and verifies monotonicity.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopTelemetry {
    last_spike_timestamp: Option<u64>,
}

impl Telemetry for NoopTelemetry {
    fn on_spike(&mut self, _neuron_id: usize, timestamp: u64) {
        if let Some(prev) = self.last_spike_timestamp {
            debug_assert!(
                timestamp >= prev,
                "telemetry spike timestamps must be monotonic",
            );
        }
        self.last_spike_timestamp = Some(timestamp);
    }
}
