// SPDX-License-Identifier: AGPL-3.0-only
//! Morse alphabet demo: encodes and processes all 26 letters through Cricket-Brain.

use cricket_brain::brain::CricketBrain;
use cricket_brain::patterns::encode_morse;

fn main() {
    println!("=== Cricket-Brain: Full Morse Alphabet ===\n");
    println!("{:<6} {:>8} {:>8} {:>8}", "Char", "Steps", "Spikes", "Rate");
    println!("{}", "-".repeat(36));

    for ch in 'A'..='Z' {
        let mut brain = CricketBrain::new(Default::default()).expect("valid default brain config");
        let signal = encode_morse(&ch.to_string());

        let mut total_steps = 0;
        let mut spike_count = 0;

        for &(freq, duration) in &signal {
            for _ in 0..duration {
                let output = brain.step(freq);
                total_steps += 1;
                if output > 0.0 {
                    spike_count += 1;
                }
            }
        }

        let rate = if total_steps > 0 {
            spike_count as f64 / total_steps as f64 * 100.0
        } else {
            0.0
        };

        println!("{ch:<6} {total_steps:>8} {spike_count:>8} {rate:>7.1}%");
    }
}
