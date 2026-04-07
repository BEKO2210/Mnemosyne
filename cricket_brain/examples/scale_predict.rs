// SPDX-License-Identifier: AGPL-3.0-only
//! v0.3 Scale demo: 40k-neuron sequence predictor benchmark.
//!
//! Creates a large vocabulary, registers many patterns, and measures
//! prediction throughput — proving the architecture scales linearly.

use cricket_brain::sequence::SequencePredictor;
use cricket_brain::token::TokenVocabulary;
use std::time::Instant;

fn main() {
    println!("=== Cricket-Brain v0.3: 40k Neuron Sequence Predictor ===\n");

    // Create a large vocabulary (256 tokens)
    let labels: Vec<String> = (0..256).map(|i| format!("T{i:03}")).collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

    let t0 = Instant::now();
    let vocab = TokenVocabulary::from_labels(&label_refs);
    let mut pred = SequencePredictor::with_params(vocab, 8, 300).expect("valid predictor config");

    // Register 1000 random-ish patterns (3-8 tokens each)
    for i in 0..1000 {
        let len = 3 + (i % 6);
        let pattern_labels: Vec<&str> = (0..len)
            .map(|j| label_refs[(i * 7 + j * 13) % 256])
            .collect();
        let name = format!("pat_{i:04}");
        pred.register_pattern(&name, &pattern_labels)
            .expect("generated pattern labels must exist in vocabulary");
    }

    let init_time = t0.elapsed();
    let neurons = pred.total_neurons();
    let mem = pred.memory_usage_bytes();

    println!("Vocabulary:       {} tokens", label_refs.len());
    println!("Patterns:         {}", pred.patterns.len());
    println!("Total neurons:    {neurons}");
    println!("Init time:        {init_time:.2?}");
    println!("Memory:           {:.2} MB", mem as f64 / 1_048_576.0);

    // Benchmark: feed 1000 steps
    let steps = 1000;
    let freq = pred.bank.channels[0].neurons[0].eigenfreq;
    let t1 = Instant::now();
    for _ in 0..steps {
        pred.step(freq);
    }
    let run_time = t1.elapsed();
    let steps_per_sec = steps as f64 / run_time.as_secs_f64();
    let neuron_ops = steps_per_sec * neurons as f64;

    println!("\n--- Throughput ---");
    println!("Steps:            {steps}");
    println!("Time:             {run_time:.2?}");
    println!("Steps/sec:        {steps_per_sec:.0}");
    println!("Neuron-ops/sec:   {neuron_ops:.2e}");

    // Check predictions
    let prediction = pred.predict();
    println!("\n--- Prediction State ---");
    println!("Active matchers:  {}", pred.active_matchers());
    match prediction {
        Some(p) => println!(
            "Best prediction:  '{}' (pattern: \"{}\", conf: {:.2})",
            p.label, p.pattern_name, p.confidence
        ),
        None => println!("Best prediction:  (none active)"),
    }

    println!("\n--- Comparison ---");
    println!(
        "Cricket-Brain v0.3: {neurons} neurons, {:.2} MB, {neuron_ops:.2e} ops/sec",
        mem as f64 / 1_048_576.0
    );
    println!("GPT-4:              ~1.8T params, ~800 GB, ~3.1e17 FLOPS");
    println!("\nCricket-Brain does sequence prediction with:");
    println!("  - 0 training");
    println!("  - 0 GPU");
    println!("  - {:.2} MB RAM", mem as f64 / 1_048_576.0);
    println!("  - Pattern registration instead of gradient descent");
}
