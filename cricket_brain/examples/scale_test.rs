// SPDX-License-Identifier: AGPL-3.0-only
//! Scale test: Creates a 40960-neuron CricketBrain and benchmarks throughput.

use cricket_brain::brain::CricketBrain;
use std::time::Instant;

fn main() {
    println!("=== Cricket-Brain: Scale Test (40960 Neurons) ===\n");

    // Initialization benchmark
    let t0 = Instant::now();
    let mut brain = CricketBrain::new_scaled(40960, 40960 * 3).expect("valid scaled config");
    let init_time = t0.elapsed();

    let mem = brain.memory_usage_bytes();
    println!("Neurons:          40,960");
    println!("Synapses:         {}", 40960 * 3);
    println!("Init time:        {:.2?}", init_time);
    println!("Memory usage:     {:.2} MB", mem as f64 / 1_048_576.0);

    // Throughput benchmark: 1000 steps
    let steps = 1000;
    let t1 = Instant::now();
    for i in 0..steps {
        let freq = if i % 2 == 0 { 4500.0 } else { 0.0 };
        brain.step(freq);
    }
    let run_time = t1.elapsed();

    let steps_per_sec = steps as f64 / run_time.as_secs_f64();
    let neurons_per_sec = steps_per_sec * 40960.0;

    println!("\n--- Throughput ---");
    println!("Steps:            {steps}");
    println!("Total time:       {run_time:.2?}");
    println!("Steps/sec:        {steps_per_sec:.0}");
    println!("Neuron-steps/sec: {neurons_per_sec:.2e}");

    println!("\n--- Comparison ---");
    println!("Cricket-Brain:    {neurons_per_sec:.2e} neuron-ops/sec (single-threaded, CPU)");
    println!("GPT-4 estimate:   ~3.1e17 FLOPS (multi-GPU cluster)");
    println!(
        "Ratio:            Cricket runs at {:.2e}x the ops",
        neurons_per_sec / 3.1e17
    );
    println!(
        "\nBut Cricket-Brain needs: 0 GPU, {:.2} MB RAM, 0 training",
        mem as f64 / 1_048_576.0
    );
}
