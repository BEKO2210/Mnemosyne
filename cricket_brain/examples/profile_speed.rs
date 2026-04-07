// SPDX-License-Identifier: AGPL-3.0-only
use cricket_brain::brain::{BrainConfig, CricketBrain};
#[cfg(feature = "telemetry")]
use cricket_brain::logger::{NoopTelemetry, Telemetry};
use std::time::Instant;

const STEPS: usize = 1_000_000;
const INPUT_FREQ: f32 = 4_500.0;

fn run_loop(brain: &mut CricketBrain) -> f64 {
    let started = Instant::now();
    for _ in 0..STEPS {
        let _ = brain.step(INPUT_FREQ);
    }
    started.elapsed().as_secs_f64()
}

#[cfg(feature = "telemetry")]
fn run_loop_with_telemetry(brain: &mut CricketBrain) -> f64 {
    let mut telemetry = NoopTelemetry::default();
    let started = Instant::now();
    for _ in 0..STEPS {
        let out = brain.step(INPUT_FREQ);
        telemetry.on_resonance_change(0, brain.neurons[0].resonance_level());
        if out > 0.0 {
            telemetry.on_spike(brain.neurons.len() - 1, brain.time_step as u64);
        }
    }
    started.elapsed().as_secs_f64()
}

fn main() {
    let config = BrainConfig::default();

    let mut baseline = CricketBrain::new(config.clone()).expect("default config must be valid");
    let baseline_secs = run_loop(&mut baseline);
    let baseline_us_per_step = (baseline_secs * 1_000_000.0) / STEPS as f64;
    println!("Baseline: {baseline_us_per_step:.6} μs/step ({STEPS} steps)");

    #[cfg(feature = "telemetry")]
    {
        let mut observed = CricketBrain::new(config).expect("default config must be valid");
        let observed_secs = run_loop_with_telemetry(&mut observed);
        let observed_us_per_step = (observed_secs * 1_000_000.0) / STEPS as f64;
        let overhead_pct = ((observed_us_per_step / baseline_us_per_step) - 1.0) * 100.0;
        println!("Telemetry: {observed_us_per_step:.6} μs/step ({STEPS} steps)");
        println!("Observer effect: {overhead_pct:.2}%");
    }

    #[cfg(not(feature = "telemetry"))]
    {
        println!("Telemetry profile disabled. Re-run with: cargo run --example profile_speed --features telemetry");
    }
}
