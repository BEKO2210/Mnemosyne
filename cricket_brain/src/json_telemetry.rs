// SPDX-License-Identifier: AGPL-3.0-only
use crate::logger::{Telemetry, TelemetryEvent};
use serde::Serialize;
use std::io::{BufWriter, Write};

/// JSON Lines (JSONL) telemetry sink that serializes [`TelemetryEvent`]s to a
/// writer, one JSON object per line. Suitable for piping into dashboards,
/// log aggregators, or file-based analysis pipelines.
#[derive(Debug)]
pub struct JsonTelemetry<W: Write> {
    sink: BufWriter<W>,
}

#[derive(Serialize)]
struct JsonEvent<'a> {
    #[serde(rename = "type")]
    event_type: &'a str,
    neuron_id: Option<usize>,
    timestamp: Option<u64>,
    value: Option<f32>,
    pattern_id: Option<usize>,
    ratio: Option<f32>,
    confidence: Option<f32>,
    jitter: Option<f32>,
    tolerance: Option<f32>,
    active_neurons: Option<usize>,
    total_neurons: Option<usize>,
}

impl<W: Write> JsonTelemetry<W> {
    /// Creates a new JSON telemetry sink writing to the given [`Write`] target.
    pub fn new(writer: W) -> Self {
        Self {
            sink: BufWriter::new(writer),
        }
    }

    /// Flushes the internal buffer, ensuring all queued events are written out.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.sink.flush()
    }

    fn write_event(&mut self, event: JsonEvent<'_>) {
        if serde_json::to_writer(&mut self.sink, &event).is_ok() {
            let _ = self.sink.write_all(b"\n");
        }
    }
}

impl JsonTelemetry<std::io::Stdout> {
    /// Convenience constructor that writes JSONL events to standard output.
    pub fn stdout() -> Self {
        Self::new(std::io::stdout())
    }
}

impl<W: Write> Telemetry for JsonTelemetry<W> {
    fn on_event(&mut self, event: TelemetryEvent) {
        match event {
            TelemetryEvent::Spike {
                neuron_id,
                timestamp,
            } => self.write_event(JsonEvent {
                event_type: "spike",
                neuron_id: Some(neuron_id),
                timestamp: Some(timestamp),
                value: None,
                pattern_id: None,
                ratio: None,
                confidence: None,
                jitter: None,
                tolerance: None,
                active_neurons: None,
                total_neurons: None,
            }),
            TelemetryEvent::ResonanceChange { neuron_id, value } => self.write_event(JsonEvent {
                event_type: "resonance",
                neuron_id: Some(neuron_id),
                timestamp: None,
                value: Some(value),
                pattern_id: None,
                ratio: None,
                confidence: None,
                jitter: None,
                tolerance: None,
                active_neurons: None,
                total_neurons: None,
            }),
            TelemetryEvent::SequenceMatched {
                pattern_id,
                confidence,
                snr,
                jitter,
                tolerance,
            } => self.write_event(JsonEvent {
                event_type: "sequence_match",
                neuron_id: None,
                timestamp: None,
                value: None,
                pattern_id: Some(pattern_id),
                ratio: Some(snr),
                confidence: Some(confidence),
                jitter: Some(jitter),
                tolerance: Some(tolerance),
                active_neurons: None,
                total_neurons: None,
            }),
            TelemetryEvent::SnrReport { ratio } => self.write_event(JsonEvent {
                event_type: "snr",
                neuron_id: None,
                timestamp: None,
                value: None,
                pattern_id: None,
                ratio: Some(ratio),
                confidence: None,
                jitter: None,
                tolerance: None,
                active_neurons: None,
                total_neurons: None,
            }),
            TelemetryEvent::SystemOverload {
                entropy,
                active_neurons,
                total_neurons,
            } => self.write_event(JsonEvent {
                event_type: "system_overload",
                neuron_id: None,
                timestamp: None,
                value: Some(entropy),
                pattern_id: None,
                ratio: None,
                confidence: None,
                jitter: None,
                tolerance: None,
                active_neurons: Some(active_neurons),
                total_neurons: Some(total_neurons),
            }),
        }
    }
}
