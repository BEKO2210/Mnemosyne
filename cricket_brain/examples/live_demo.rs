// SPDX-License-Identifier: AGPL-3.0-only
//! Interactive demo: Full encode → brain → decode pipeline.
//!
//! Shows how to bring the Cricket-Brain "to life" — feeding it a message,
//! collecting spike output, and decoding it back to text.
//!
//! Usage:
//!   cargo run --example live_demo
//!   cargo run --example live_demo -- "HELLO WORLD"

use cricket_brain::brain::CricketBrain;
use cricket_brain::patterns::{decode_spikes, encode_morse};

fn main() {
    let message = std::env::args().nth(1).unwrap_or_else(|| "SOS".to_string());

    println!("=== Cricket-Brain: Live Demo ===\n");
    println!("Input message: \"{message}\"\n");

    // Step 1: Encode text to frequency signal
    let signal = encode_morse(&message);
    let total_ms: usize = signal.iter().map(|&(_, d)| d).sum();
    println!(
        "Encoded: {} segments, {} ms total\n",
        signal.len(),
        total_ms
    );

    // Step 2: Feed signal through the brain, collect spikes
    let mut brain = CricketBrain::new(Default::default()).expect("valid default brain config");
    let mut spikes: Vec<(usize, f32)> = Vec::new();
    let mut t = 0;

    for &(freq, duration) in &signal {
        for _ in 0..duration {
            let output = brain.step(freq);
            spikes.push((t, output));
            t += 1;
        }
    }

    // Step 3: Visualize the spike train
    println!("--- Spike Train (each char = 10ms) ---");
    let mut line = String::new();
    let mut spike_count = 0;
    for chunk in spikes.chunks(10) {
        let has_spike = chunk.iter().any(|&(_, amp)| amp > 0.0);
        if has_spike {
            line.push('|');
            spike_count += chunk.iter().filter(|&&(_, a)| a > 0.0).count();
        } else {
            line.push('_');
        }
    }
    println!("{line}");
    println!("({spike_count} spikes in {t} ms)\n");

    // Step 4: Decode spikes back to text
    let decoded = decode_spikes(&spikes, 0.1);
    println!("Decoded output: \"{decoded}\"");
    println!(
        "Match: {}",
        if decoded.trim() == message.to_uppercase().trim() {
            "EXACT MATCH"
        } else {
            "PARTIAL (spike-timing decode is lossy — see docs)"
        }
    );

    // Step 5: Show per-neuron state
    println!("\n--- Final Neuron States ---");
    let names = ["AN1", "LN2", "LN3", "LN5", "ON1"];
    for (i, neuron) in brain.neurons.iter().enumerate() {
        println!(
            "  {} (id={}): amplitude={:.4}, phase={:.4}, eigenfreq={:.0} Hz",
            names.get(i).unwrap_or(&"???"),
            neuron.id,
            neuron.amplitude,
            neuron.phase,
            neuron.eigenfreq,
        );
    }

    println!("\nMemory: {} bytes", brain.memory_usage_bytes());
    println!("\nTip: Try different messages:");
    println!("  cargo run --example live_demo -- \"HELLO\"");
    println!("  cargo run --example live_demo -- \"CQ CQ CQ\"");
    println!("  cargo run --example live_demo -- \"73\"");
}
