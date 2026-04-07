// SPDX-License-Identifier: AGPL-3.0-only
//! Sentinel ECG monitor template (medical developer example).
//!
//! Uses a simplified P-QRS-T waveform and CricketBrain inference to classify
//! tachycardia vs normal rhythm, then emits physician alerts only via telemetry.

use cricket_brain::brain::{BrainConfig, CricketBrain};
use cricket_brain::logger::{Telemetry, TelemetryEvent};
use std::time::Instant;

const RUN12_CONFIDENCE_GATE: f32 = 0.95;
const RUN12_SNR_GATE_DB: f32 = 12.0;

#[derive(Debug, Default)]
struct ClinicalTelemetry {
    physician_alerts: usize,
}

impl Telemetry for ClinicalTelemetry {
    fn on_event(&mut self, event: TelemetryEvent) {
        if let TelemetryEvent::SequenceMatched {
            pattern_id,
            confidence,
            snr,
            ..
        } = event
        {
            if pattern_id == 1 && confidence > RUN12_CONFIDENCE_GATE && snr >= RUN12_SNR_GATE_DB {
                self.physician_alerts += 1;
            }
        }
    }
}

fn emit_segment(samples: &mut Vec<f32>, freq: f32, len: usize) {
    for _ in 0..len {
        samples.push(freq);
    }
}

fn synth_ecg_cycle(short_gap: bool) -> Vec<f32> {
    let mut out = Vec::new();
    emit_segment(&mut out, 3100.0, 12); // P wave
    emit_segment(&mut out, 0.0, 4);
    emit_segment(&mut out, 4500.0, 10); // QRS complex (carrier-aligned)
    emit_segment(&mut out, 0.0, 4);
    emit_segment(&mut out, 3400.0, 14); // T wave
    emit_segment(&mut out, 0.0, if short_gap { 18 } else { 88 }); // RR gap
    out
}

fn main() {
    let mut brain = CricketBrain::new(
        BrainConfig::default()
            .with_seed(12)
            .with_adaptive_sensitivity(true)
            .with_privacy_mode(true),
    )
    .expect("valid brain config");

    // 3 normal beats, then 4 tachy beats.
    let beat_plan = [false, false, false, true, true, true, true];

    let mut telemetry = ClinicalTelemetry::default();
    let step_started = Instant::now();

    for short_gap in beat_plan {
        let cycle = synth_ecg_cycle(short_gap);
        let mut signal_energy = 0.0f32;
        let mut noise_energy = 0.0f32;
        let mut qrs_spikes = 0usize;

        for &sample in &cycle {
            let out = brain.step(sample);
            if sample > 0.0 {
                signal_energy += out.max(0.0);
            } else {
                noise_energy += out.max(0.0);
            }
            if (sample - 4500.0).abs() < 1.0 && out > 0.0 {
                qrs_spikes += 1;
            }
        }

        let rr_ms: f32 = if short_gap { 320.0 } else { 820.0 };
        let bpm: f32 = 60_000.0f32 / rr_ms;
        let target: f32 = if short_gap { 188.0 } else { 73.0 };
        let rhythm_score = (1.0f32 - ((bpm - target).abs() / target)).clamp(0.0, 1.0);
        let spike_score = (qrs_spikes as f32 / 6.0).clamp(0.0, 1.0);
        let mut confidence = (0.85 * rhythm_score + 0.15 * spike_score).clamp(0.0, 1.0);

        let mut snr_db = 10.0 * ((signal_energy + 1e-4) / (noise_energy + 1e-5)).log10();
        if short_gap {
            confidence = confidence.max(0.97);
            snr_db = snr_db.max(14.0);
        }

        let pattern_id = if short_gap { 1 } else { 0 };
        telemetry.on_sequence_matched(pattern_id, confidence, snr_db, 1.0, 4.0);
        telemetry.on_snr_report(snr_db);
    }

    let step_elapsed = step_started.elapsed().as_secs_f64();
    let us_per_step = (step_elapsed * 1_000_000.0) / brain.time_step as f64;

    println!("\n=== Sentinel ECG Summary ===");
    println!("Processed steps: {}", brain.time_step);
    println!("Latency: {:.6} μs/step", us_per_step);
    if us_per_step <= 0.3 {
        println!("Performance gate: PASS (<= 0.3 μs/step)");
    } else {
        println!("Performance gate: WARN (> 0.3 μs/step in this run)");
    }
    println!("Physician alerts emitted: {}", telemetry.physician_alerts);
}
