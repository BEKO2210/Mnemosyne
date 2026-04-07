// SPDX-License-Identifier: AGPL-3.0-only
//! Frequency discrimination demo: proves the brain rejects wrong frequencies.
//!
//! The cricket brain is tuned to 4500 Hz. This demo shows that signals
//! at other frequencies produce zero or near-zero output — the core
//! value proposition of the neuromorphic approach.

use cricket_brain::brain::CricketBrain;

fn main() {
    println!("=== Cricket-Brain: Frequency Discrimination ===\n");
    println!(
        "{:<12} {:>8} {:>8} {:>10}",
        "Frequency", "Spikes", "Steps", "Spike Rate"
    );
    println!("{}", "-".repeat(42));

    let test_freqs = [
        1000.0, 2000.0, 3000.0, 3500.0, 4000.0, 4200.0, 4400.0, 4500.0, 4600.0, 4800.0, 5000.0,
        5500.0, 6000.0, 7000.0, 8000.0,
    ];

    for &freq in &test_freqs {
        let mut brain = CricketBrain::new(Default::default()).expect("valid default brain config");
        let steps = 200;
        let mut spikes = 0;

        for _ in 0..steps {
            let output = brain.step(freq);
            if output > 0.0 {
                spikes += 1;
            }
        }

        let rate = spikes as f64 / steps as f64 * 100.0;
        let bar = "|".repeat((rate / 2.0) as usize);
        println!(
            "{freq:>8.0} Hz {:>8} {:>8} {:>8.1}%  {bar}",
            spikes, steps, rate
        );
    }

    println!("\nThe brain is tuned to 4500 Hz (cricket carrier frequency).");
    println!("Frequencies outside the ~10% Gaussian window produce zero spikes.");
    println!("This is the biological equivalent of a bandpass filter — but with");
    println!("temporal coincidence detection, not just frequency matching.");
}
