// SPDX-License-Identifier: AGPL-3.0-only
//! Morse Code "SOS" recognition demo using Cricket-Brain.
//!
//! Demonstrates the biomorphic inference engine detecting a Morse-coded
//! SOS signal using delay-line coincidence detection.

use cricket_brain::brain::CricketBrain;
use cricket_brain::patterns::{encode_morse, MORSE_FREQ};

fn main() {
    println!("=== Cricket-Brain: Morse SOS Demo ===\n");

    let mut brain = CricketBrain::new(Default::default()).expect("valid default brain config");
    let signal = encode_morse("SOS");

    println!(
        "Signal: {} segments, carrier = {} Hz\n",
        signal.len(),
        MORSE_FREQ
    );

    let mut total_steps = 0;
    let mut spike_count = 0;
    let mut silence_spikes = 0;

    for &(freq, duration) in &signal {
        let label = if freq > 0.0 { "TONE" } else { "PAUSE" };
        let mut segment_spikes = 0;

        for _ in 0..duration {
            let output = brain.step(freq);
            total_steps += 1;

            if output > 0.0 {
                segment_spikes += 1;
                spike_count += 1;
                if freq == 0.0 {
                    silence_spikes += 1;
                }
            }
        }

        if segment_spikes > 0 {
            println!("  {label:>5} {duration:>4}ms @ {freq:>6.0} Hz → {segment_spikes} spikes");
        }
    }

    println!("\n--- Summary ---");
    println!("Total timesteps: {total_steps}");
    println!("Total spikes:    {spike_count}");
    println!("Silence spikes:  {silence_spikes} (should be 0)");
    println!("Memory usage:    {} bytes", brain.memory_usage_bytes());
    println!(
        "\n{}",
        if silence_spikes == 0 {
            "✓ PASS: No false positives during silence"
        } else {
            "✗ FAIL: Spikes detected during silence!"
        }
    );
}
