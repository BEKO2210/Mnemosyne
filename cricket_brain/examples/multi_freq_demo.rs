// SPDX-License-Identifier: AGPL-3.0-only
//! v0.2 Demo: Multi-frequency token recognition.
//!
//! Encodes each character as a unique frequency and shows that
//! parallel resonator banks can discriminate between tokens in real-time.
//!
//! Usage:
//!   cargo run --example multi_freq_demo
//!   cargo run --example multi_freq_demo -- "RUST"

use cricket_brain::resonator_bank::ResonatorBank;
use cricket_brain::token::TokenVocabulary;

fn main() {
    let message = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "HELLO".to_string());

    println!("=== Cricket-Brain v0.2: Multi-Frequency Token Recognition ===\n");

    let vocab = TokenVocabulary::from_alphabet();
    let mut bank = ResonatorBank::new(&vocab);

    println!("Vocabulary: {} tokens (A-Z + space)", vocab.len());
    println!(
        "Frequency band: {:.0}–{:.0} Hz, spacing: {:.1} Hz/token",
        vocab.freq_min,
        vocab.freq_max,
        vocab.freq_spacing()
    );
    println!(
        "Resonator bank: {} neurons, {} synapses\n",
        bank.total_neurons(),
        bank.total_synapses()
    );

    // Encode the message as multi-frequency tokens
    let signal = vocab.encode_text(&message, 60, 40);
    let upper = message.to_uppercase();
    let chars: Vec<char> = upper.chars().collect();

    println!("Input: \"{upper}\"\n");
    println!(
        "{:<6} {:>8} {:>10} {:>10} {:>8}",
        "Char", "Freq Hz", "Spikes", "Detected", "Correct"
    );
    println!("{}", "-".repeat(48));

    let mut total_correct = 0;
    let mut total_chars = 0;
    let mut char_idx = 0;

    for &(freq, duration) in &signal {
        if freq <= 0.0 {
            // Silence gap — just process
            for _ in 0..duration {
                bank.step(0.0);
            }
            continue;
        }

        // This is a token signal
        let expected_char = if char_idx < chars.len() {
            chars[char_idx]
        } else {
            '?'
        };

        let mut spike_counts = vec![0usize; vocab.len()];
        for _ in 0..duration {
            let activations = bank.step(freq);
            for (i, &a) in activations.iter().enumerate() {
                if a > 0.0 {
                    spike_counts[i] += 1;
                }
            }
        }

        // Find which token fired the most
        let (best_id, &best_count) = spike_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .unwrap();

        let detected_label = vocab
            .get_by_id(best_id)
            .map(|t| t.label.as_str())
            .unwrap_or("?");

        let is_correct = detected_label == expected_char.to_string();
        if is_correct {
            total_correct += 1;
        }
        total_chars += 1;

        let status = if is_correct { "OK" } else { "MISS" };

        println!(
            "{:<6} {:>8.0} {:>10} {:>10} {:>8}",
            expected_char, freq, best_count, detected_label, status
        );

        char_idx += 1;
    }

    let accuracy = if total_chars > 0 {
        total_correct as f64 / total_chars as f64 * 100.0
    } else {
        0.0
    };

    println!("\n--- Summary ---");
    println!("Characters: {total_chars}");
    println!("Correct:    {total_correct}");
    println!("Accuracy:   {accuracy:.1}%");
    println!("Memory:     {} bytes", bank.memory_usage_bytes());
    println!(
        "\n{}",
        if accuracy == 100.0 {
            "PASS: Perfect multi-frequency discrimination!"
        } else {
            "PARTIAL: Some tokens confused (frequencies too close)"
        }
    );
}
