// SPDX-License-Identifier: AGPL-3.0-only
//! v0.3 Demo: Sequence prediction using delay-line pattern memory.
//!
//! Registers word patterns, feeds partial sequences, and predicts the next token.
//! No training, no gradient descent — just topological pattern matching.
//!
//! Usage:
//!   cargo run --example sequence_predict

use cricket_brain::sequence::SequencePredictor;
use cricket_brain::token::TokenVocabulary;

fn feed_token(pred: &mut SequencePredictor, label: &str, hold_ms: usize, gap_ms: usize) {
    let freq = pred.vocab.get(label).unwrap().freq;
    for _ in 0..hold_ms {
        pred.step(freq);
    }
    for _ in 0..gap_ms {
        pred.step(0.0);
    }
}

fn main() {
    println!("=== Cricket-Brain v0.3: Sequence Prediction ===\n");

    // Use a well-spaced vocabulary with only the tokens we need.
    // 8 tokens across 2000–9000 Hz = 1000 Hz spacing (clean Gaussian separation).
    let vocab = TokenVocabulary::new(&["H", "E", "L", "O", "P", "S", "W", "R"], 2000.0, 9000.0);

    let mut pred = SequencePredictor::with_params(vocab, 8, 300).expect("valid predictor config");

    // Register known patterns (our "training data" — but it's topology, not weights)
    pred.register_pattern("hello", &["H", "E", "L", "L", "O"])
        .expect("pattern tokens must exist");
    pred.register_pattern("help", &["H", "E", "L", "P"])
        .expect("pattern tokens must exist");
    pred.register_pattern("world", &["W", "O", "R", "L"])
        .expect("pattern tokens must exist");
    pred.register_pattern("sos", &["S", "O", "S"])
        .expect("pattern tokens must exist");

    println!(
        "Registered {} patterns: {}",
        pred.patterns.len(),
        pred.patterns
            .iter()
            .map(|p| format!("\"{}\"", p.name))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Vocabulary: {} tokens, {:.0} Hz spacing",
        pred.bank.channels.len(),
        pred.vocab.freq_spacing()
    );
    println!("Total neurons: {}\n", pred.total_neurons(),);

    // === Test 1: "H, E, L" → should predict "L" (hello) or "P" (help) ===
    println!("--- Test 1: Feed H → E → L (shared prefix of 'hello' and 'help') ---");
    feed_token(&mut pred, "H", 60, 50);
    print_state(&pred, "H");

    feed_token(&mut pred, "E", 60, 50);
    print_state(&pred, "H,E");

    feed_token(&mut pred, "L", 60, 50);
    print_state(&pred, "H,E,L");

    // === Test 2: Continue with "L" → disambiguates to "hello" ===
    println!("\n--- Test 2: Continue with L → disambiguates to 'hello' ---");
    feed_token(&mut pred, "L", 60, 50);
    print_state(&pred, "H,E,L,L");

    // === Test 3: Reset and test SOS ===
    pred.reset();
    println!("\n--- Test 3: Feed S → O (pattern: 'sos') ---");
    feed_token(&mut pred, "S", 60, 50);
    print_state(&pred, "S");

    feed_token(&mut pred, "O", 60, 50);
    print_state(&pred, "S,O");

    // === Test 4: Show all predictions for ambiguous input ===
    pred.reset();
    println!("\n--- Test 4: Feed W → O (could be 'world') ---");
    feed_token(&mut pred, "W", 60, 50);
    print_state(&pred, "W");

    feed_token(&mut pred, "O", 60, 50);
    print_state(&pred, "W,O");

    // === Summary ===
    println!("\n--- Summary ---");
    println!("Neurons:          {}", pred.total_neurons());
    println!("Patterns stored:  {}", pred.patterns.len());
    println!("Memory:           {} bytes", pred.memory_usage_bytes());
    println!("History:          {:?}", pred.history_labels());
    println!("\nKey insight: Patterns are stored as TOPOLOGY, not weights.");
    println!("Prediction = delay-line coincidence across token detections.");
    println!("No gradient descent. No backprop. No GPU. No training loop.");
}

fn print_state(pred: &SequencePredictor, context: &str) {
    let all = pred.predict_all();
    if all.is_empty() {
        println!("  [{context}] → (no active pattern)");
        return;
    }

    for (i, p) in all.iter().enumerate() {
        let pattern_len = pred
            .patterns
            .iter()
            .find(|pat| pat.name == p.pattern_name)
            .map(|pat| pat.token_ids.len())
            .unwrap_or(0);

        let marker = if i == 0 { "→" } else { " " };
        println!(
            "  [{context}] {marker} predict '{}' (pattern: \"{}\", progress: {}/{}, conf: {:.2})",
            p.label, p.pattern_name, p.matched_length, pattern_len, p.confidence
        );
    }
}
