// SPDX-License-Identifier: AGPL-3.0-only
//! Research data generator for "Silent Sentinel" whitepaper prep.
//!
//! Generates:
//! - SNR sweep detection-rate curves (-10 dB..+30 dB)
//! - ROC-like operating points (TPR/FPR) over sensitivity settings
//! - CSV + JSON artifacts suitable for plotting or statistical post-processing
//!
//! Headless mode (default): no UI, no interactive prompts.
//!
//! Usage:
//! ```bash
//! cargo run --release --example research_gen -- --output ./target/research --seed 1337
//! ```

use cricket_brain::brain::{BrainConfig, CricketBrain};
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }

    fn centered(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

#[derive(Debug, Clone)]
struct SweepResult {
    snr_db: i32,
    sensitivity: f32,
    tp: usize,
    fp: usize,
    tn: usize,
    fnn: usize,
}

impl SweepResult {
    fn tpr(&self) -> f32 {
        let denom = (self.tp + self.fnn).max(1) as f32;
        self.tp as f32 / denom
    }

    fn fpr(&self) -> f32 {
        let denom = (self.fp + self.tn).max(1) as f32;
        self.fp as f32 / denom
    }

    /// Wilson score 95% confidence interval for a proportion.
    /// More accurate than the normal approximation for small N or extreme p.
    fn wilson_ci(successes: usize, total: usize) -> (f32, f32) {
        if total == 0 {
            return (0.0, 1.0);
        }
        let n = total as f64;
        let p = successes as f64 / n;
        let z = 1.96_f64; // 95% CI
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denom;
        let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denom;
        (
            (center - margin).max(0.0) as f32,
            (center + margin).min(1.0) as f32,
        )
    }

    fn tpr_ci(&self) -> (f32, f32) {
        Self::wilson_ci(self.tp, self.tp + self.fnn)
    }

    fn fpr_ci(&self) -> (f32, f32) {
        Self::wilson_ci(self.fp, self.fp + self.tn)
    }
}

fn signal_present_freq(rng: &mut Lcg, snr_db: i32) -> f32 {
    let noise_scale = 10.0_f32.powf(-snr_db as f32 / 20.0);
    let jitter_hz = 180.0 * noise_scale * rng.centered();
    (4500.0 + jitter_hz).clamp(2000.0, 8000.0)
}

fn background_freq(rng: &mut Lcg, snr_db: i32) -> f32 {
    // As SNR gets worse, random distractors become more likely and stronger.
    let noise_scale = 10.0_f32.powf(-snr_db as f32 / 20.0).min(4.0);
    let burst_prob = (0.03 * noise_scale).clamp(0.01, 0.18);
    if rng.next_f32() < burst_prob {
        2000.0 + rng.next_f32() * 6000.0
    } else {
        0.0
    }
}

fn run_trial(brain: &mut CricketBrain, rng: &mut Lcg, snr_db: i32, target_present: bool) -> bool {
    brain.reset();

    // Warm-up silence.
    for _ in 0..24 {
        let _ = brain.step(background_freq(rng, snr_db));
    }

    // Observation window.
    let mut detected = false;
    for t in 0..120 {
        let freq = if target_present && (32..92).contains(&t) {
            signal_present_freq(rng, snr_db)
        } else {
            background_freq(rng, snr_db)
        };
        let out = brain.step(freq);
        if out > 0.0 {
            detected = true;
        }
    }

    detected
}

fn parse_arg(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find_map(|w| (w[0] == key).then(|| w[1].clone()))
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let output_dir = parse_arg(&args, "--output", "target/research");
    let seed: u64 = parse_arg(&args, "--seed", "1337").parse().unwrap_or(1337);
    let _headless = args.iter().any(|a| a == "--headless");

    let output_path = PathBuf::from(output_dir);
    fs::create_dir_all(&output_path).expect("create output dir");

    let snr_levels: Vec<i32> = (-10..=30).step_by(5).collect();
    let sensitivity_grid: [f32; 8] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90];
    let trials_per_class = 500usize;

    let mut all = Vec::<SweepResult>::new();

    for &sens in &sensitivity_grid {
        let mut rng = Lcg::new(seed ^ (sens.to_bits() as u64));
        let cfg = BrainConfig::default()
            .with_adaptive_sensitivity(true)
            .with_min_activation_threshold(sens)
            .with_seed(seed ^ (sens.to_bits() as u64));
        let mut brain = CricketBrain::new(cfg).expect("valid brain config");

        for &snr_db in &snr_levels {
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut tn = 0usize;
            let mut fnn = 0usize;

            for _ in 0..trials_per_class {
                let detected = run_trial(&mut brain, &mut rng, snr_db, true);
                if detected {
                    tp += 1;
                } else {
                    fnn += 1;
                }
            }

            for _ in 0..trials_per_class {
                let detected = run_trial(&mut brain, &mut rng, snr_db, false);
                if detected {
                    fp += 1;
                } else {
                    tn += 1;
                }
            }

            all.push(SweepResult {
                snr_db,
                sensitivity: sens,
                tp,
                fp,
                tn,
                fnn,
            });
        }
    }

    let csv_path = output_path.join("sentinel_sweep.csv");
    let json_path = output_path.join("sentinel_sweep.json");

    let mut csv = String::from(
        "snr_db,sensitivity,tp,fp,tn,fn,tpr,tpr_ci_lo,tpr_ci_hi,fpr,fpr_ci_lo,fpr_ci_hi\n",
    );
    for r in &all {
        let (tpr_lo, tpr_hi) = r.tpr_ci();
        let (fpr_lo, fpr_hi) = r.fpr_ci();
        csv.push_str(&format!(
            "{},{:.2},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            r.snr_db,
            r.sensitivity,
            r.tp,
            r.fp,
            r.tn,
            r.fnn,
            r.tpr(),
            tpr_lo,
            tpr_hi,
            r.fpr(),
            fpr_lo,
            fpr_hi,
        ));
    }
    fs::write(&csv_path, csv).expect("write csv");

    let mut json = String::from("{\n  \"seed\": ");
    json.push_str(&seed.to_string());
    json.push_str(
        ",\n  \"snr_range_db\": [-10, 30],\n  \"snr_step_db\": 5,\n  \"trials_per_class\": ",
    );
    json.push_str(&trials_per_class.to_string());
    json.push_str(",\n  \"results\": [\n");
    for (idx, r) in all.iter().enumerate() {
        let comma = if idx + 1 == all.len() { "" } else { "," };
        let (tpr_lo, tpr_hi) = r.tpr_ci();
        let (fpr_lo, fpr_hi) = r.fpr_ci();
        json.push_str(&format!(
            "    {{\"snr_db\":{},\"sensitivity\":{:.2},\"tp\":{},\"fp\":{},\"tn\":{},\"fn\":{},\"tpr\":{:.6},\"tpr_ci\":[{:.6},{:.6}],\"fpr\":{:.6},\"fpr_ci\":[{:.6},{:.6}]}}{}\n",
            r.snr_db, r.sensitivity, r.tp, r.fp, r.tn, r.fnn,
            r.tpr(), tpr_lo, tpr_hi,
            r.fpr(), fpr_lo, fpr_hi,
            comma
        ));
    }
    json.push_str("  ]\n}\n");
    fs::write(&json_path, json).expect("write json");

    println!("Generated research artifacts:");
    println!("  - {}", csv_path.display());
    println!("  - {}", json_path.display());
    println!("Total points: {}", all.len());

    println!("\nROC sample (SNR = 0 dB) with 95% Wilson CIs:");
    for r in all.iter().filter(|r| r.snr_db == 0) {
        let (tpr_lo, tpr_hi) = r.tpr_ci();
        let (fpr_lo, fpr_hi) = r.fpr_ci();
        println!(
            "  sens={:.2} -> TPR={:.3} [{:.3}, {:.3}], FPR={:.3} [{:.3}, {:.3}]",
            r.sensitivity,
            r.tpr(),
            tpr_lo,
            tpr_hi,
            r.fpr(),
            fpr_lo,
            fpr_hi,
        );
    }
}
