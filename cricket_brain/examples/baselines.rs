// SPDX-License-Identifier: AGPL-3.0-only
//! Classical signal detection baselines for comparison with CricketBrain.
//!
//! Implements three classical detectors operating on the same test protocol
//! as `research_gen.rs` so results are directly comparable:
//!
//!   1. **Matched Filter** – cross-correlates input with a 60-sample 4500 Hz cosine template.
//!   2. **Goertzel / FFT+Threshold** – single-frequency DFT magnitude at 4500 Hz.
//!   3. **IIR Bandpass** – second-order IIR centered at 4500 Hz, bandwidth 900 Hz.
//!   4. **CricketBrain** – biomorphic baseline for direct comparison.
//!
//! Outputs:
//!   - Markdown comparison table to stdout
//!   - CSV to `target/research/baseline_comparison.csv`
//!
//! Usage:
//! ```bash
//! cargo run --release --example baselines
//! ```

use cricket_brain::brain::{BrainConfig, CricketBrain};
use std::f32::consts::PI;
use std::fs;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Deterministic LCG — identical copy from research_gen.rs
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Signal generation — identical to research_gen.rs
// ---------------------------------------------------------------------------

fn signal_present_freq(rng: &mut Lcg, snr_db: i32) -> f32 {
    let noise_scale = 10.0_f32.powf(-snr_db as f32 / 20.0);
    let jitter_hz = 180.0 * noise_scale * rng.centered();
    (4500.0 + jitter_hz).clamp(2000.0, 8000.0)
}

fn background_freq(rng: &mut Lcg, snr_db: i32) -> f32 {
    let noise_scale = 10.0_f32.powf(-snr_db as f32 / 20.0).min(4.0);
    let burst_prob = (0.03 * noise_scale).clamp(0.01, 0.18);
    if rng.next_f32() < burst_prob {
        2000.0 + rng.next_f32() * 6000.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Shared trial signal builder
//
// Returns (warmup, observation) frequency vectors matching the exact RNG
// consumption pattern of research_gen.rs:
//   - 24 warmup steps: background_freq() calls
//   - 120 observation steps: signal_present_freq() in [32..92], else background
//
// Both vectors are returned so each detector receives identical input.
// The RNG is consumed exactly once per trial — no double-draw.
// ---------------------------------------------------------------------------

fn build_trial_signal(rng: &mut Lcg, snr_db: i32, target_present: bool) -> (Vec<f32>, Vec<f32>) {
    let warmup: Vec<f32> = (0..24).map(|_| background_freq(rng, snr_db)).collect();
    let observation: Vec<f32> = (0..120)
        .map(|t| {
            if target_present && (32..92).contains(&t) {
                signal_present_freq(rng, snr_db)
            } else {
                background_freq(rng, snr_db)
            }
        })
        .collect();
    (warmup, observation)
}

// ---------------------------------------------------------------------------
// Frequency → audio sample conversion
//
// The protocol's frequency labels (e.g. 4500 Hz) are symbolic — the actual
// Fs is 1000 Sa/s (one sample per ms step).  We normalise all frequencies
// into the representable band [0, Fs/2 = 500 Hz] by mapping the protocol
// range [2000, 8000] linearly to [100, 400] Hz.  This gives every method
// (MF, Goertzel, IIR, CricketBrain) the same consistent signal, avoiding
// aliasing artefacts that would unfairly penalise frequency-domain methods.
//
// CricketBrain receives the raw protocol frequency (it has its own Gaussian
// matching that doesn't depend on Nyquist constraints).
// ---------------------------------------------------------------------------

/// Map protocol frequency (2000–8000 Hz) into representable digital band
/// (100–400 Hz @ Fs=1000).  Silence (freq ≤ 0) stays at 0.
fn map_freq(freq: f32) -> f32 {
    if freq <= 0.0 {
        0.0
    } else {
        // Linear map: 2000→100, 8000→400
        100.0 + (freq - 2000.0) * (300.0 / 6000.0)
    }
}

fn freq_to_sample(freq: f32, t: usize) -> f32 {
    let mapped = map_freq(freq);
    if mapped <= 0.0 {
        0.0
    } else {
        (2.0 * PI * mapped * t as f32 / SAMPLE_RATE).cos()
    }
}

// ---------------------------------------------------------------------------
// Baseline 1: Matched Filter
//
// Template: 60 samples of cos(2π·4500·t/1000).
// Running dot product of the last 60 input samples against the template.
// ---------------------------------------------------------------------------

const MF_LEN: usize = 60;
const TARGET_FREQ: f32 = 4500.0;
const SAMPLE_RATE: f32 = 1000.0;

fn make_mf_template() -> [f32; MF_LEN] {
    let mapped_freq = map_freq(TARGET_FREQ);
    let mut tmpl = [0.0f32; MF_LEN];
    for (i, v) in tmpl.iter_mut().enumerate() {
        *v = (2.0 * PI * mapped_freq * i as f32 / SAMPLE_RATE).cos();
    }
    tmpl
}

struct MatchedFilter {
    template: [f32; MF_LEN],
    buffer: [f32; MF_LEN],
    head: usize,
    threshold: f32,
}

impl MatchedFilter {
    fn new(threshold: f32) -> Self {
        Self {
            template: make_mf_template(),
            buffer: [0.0; MF_LEN],
            head: 0,
            threshold,
        }
    }

    /// Feed one sample; returns correlation magnitude.
    fn push(&mut self, sample: f32) -> f32 {
        self.buffer[self.head] = sample;
        self.head = (self.head + 1) % MF_LEN;

        // Dot product: buffer is a circular queue; align with template.
        let mut dot = 0.0f32;
        for k in 0..MF_LEN {
            let buf_idx = (self.head + k) % MF_LEN;
            dot += self.buffer[buf_idx] * self.template[k];
        }
        // Normalise by template length so the threshold is in [0, 1] range.
        dot / MF_LEN as f32
    }

    fn detect(&mut self, sample: f32) -> bool {
        self.push(sample).abs() > self.threshold
    }

    fn reset(&mut self) {
        self.buffer = [0.0; MF_LEN];
        self.head = 0;
    }
}

// ---------------------------------------------------------------------------
// Baseline 2: Goertzel (single-frequency DFT magnitude at 4500 Hz)
//
// Processes a rolling block of N samples and emits a magnitude each step.
// ---------------------------------------------------------------------------

struct GoertzelDetector {
    block_size: usize,
    coeff: f32,
    s1: f32,
    s2: f32,
    /// Samples accumulated in the current block.
    count: usize,
    /// Magnitude from the last completed block.
    last_magnitude: f32,
    threshold: f32,
}

impl GoertzelDetector {
    fn new(block_size: usize, threshold: f32) -> Self {
        let mapped_freq = map_freq(TARGET_FREQ);
        let coeff = 2.0 * (2.0 * PI * mapped_freq / SAMPLE_RATE).cos();
        Self {
            block_size,
            coeff,
            s1: 0.0,
            s2: 0.0,
            count: 0,
            last_magnitude: 0.0,
            threshold,
        }
    }

    /// Feed one sample; returns current magnitude estimate.
    fn push(&mut self, sample: f32) -> f32 {
        let s0 = sample + self.coeff * self.s1 - self.s2;
        self.s2 = self.s1;
        self.s1 = s0;
        self.count += 1;

        if self.count >= self.block_size {
            // Compute magnitude for completed block.
            let mag =
                (self.s1 * self.s1 + self.s2 * self.s2 - self.coeff * self.s1 * self.s2).sqrt();
            // Normalise by block size.
            self.last_magnitude = mag / self.block_size as f32;
            // Reset accumulators for the next block.
            self.s1 = 0.0;
            self.s2 = 0.0;
            self.count = 0;
        }

        self.last_magnitude
    }

    fn detect(&mut self, sample: f32) -> bool {
        self.push(sample) > self.threshold
    }

    fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
        self.count = 0;
        self.last_magnitude = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Baseline 3: Second-order IIR Bandpass Filter
//
// Design: bilinear-transform BPF using map_freq() for consistent frequency
// mapping.  All detectors now operate on the same digital signal where the
// protocol's [2000, 8000] Hz labels are mapped to [100, 400] Hz @ Fs=1000.
//
// Second-order BPF coefficients (direct form II):
//   b = [K/Q, 0, -K/Q],  a = [1 + K/Q + K², 2(K²-1)/(1+K/Q+K²), ...]
//   where K = tan(π·f_center/Fs), Q = f_center/BW.
// ---------------------------------------------------------------------------

struct IirBandpass {
    // Numerator: b0, b1, b2
    b0: f32,
    b2: f32, // b1 is always 0 for a symmetric BPF
    // Denominator (a0 = 1 after normalisation): a1, a2
    a1: f32,
    a2: f32,
    // Delay line
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
    // Envelope tracking (EMA of |y|)
    envelope: f32,
    threshold: f32,
}

impl IirBandpass {
    fn new(threshold: f32) -> Self {
        // Use the same map_freq() as all other detectors for consistency.
        // TARGET_FREQ (4500 Hz) maps to 250 Hz in the digital domain.
        let f_center = map_freq(TARGET_FREQ); // 250 Hz
        let bw = f_center * 0.2; // 20% bandwidth = 50 Hz
        let fs = SAMPLE_RATE;

        let q = f_center / bw; // Q = 5.0
        let k = (PI * f_center / fs).tan(); // tan(π·225/1000)

        let norm = 1.0 + k / q + k * k;
        let b0 = (k / q) / norm;
        let b2 = -b0;
        let a1 = (2.0 * (k * k - 1.0)) / norm;
        let a2 = (1.0 - k / q + k * k) / norm;

        Self {
            b0,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
            envelope: 0.0,
            threshold,
        }
    }

    fn push(&mut self, sample: f32) -> f32 {
        let y = self.b0 * sample + /* b1*x1 = 0 */ self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = sample;
        self.y2 = self.y1;
        self.y1 = y;
        // Smooth envelope
        self.envelope = 0.9 * self.envelope + 0.1 * y.abs();
        self.envelope
    }

    fn detect(&mut self, sample: f32) -> bool {
        self.push(sample) > self.threshold
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
        self.envelope = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Trial runner helpers
// ---------------------------------------------------------------------------

/// Result for a single trial: (detected, first_detection_step)
type TrialResult = (bool, Option<usize>);

fn run_mf_trial(mf: &mut MatchedFilter, freqs: &[f32], warmup_freqs: &[f32]) -> TrialResult {
    mf.reset();
    // Feed warmup
    for (t, &f) in warmup_freqs.iter().enumerate() {
        let s = freq_to_sample(f, t);
        mf.push(s);
    }
    let wo = warmup_freqs.len();
    let mut first = None;
    let mut detected = false;
    for (t, &f) in freqs.iter().enumerate() {
        let s = freq_to_sample(f, wo + t);
        if mf.detect(s) {
            detected = true;
            if first.is_none() {
                first = Some(t);
            }
        }
    }
    (detected, first)
}

fn run_goertzel_trial(
    gz: &mut GoertzelDetector,
    freqs: &[f32],
    warmup_freqs: &[f32],
) -> TrialResult {
    gz.reset();
    for (t, &f) in warmup_freqs.iter().enumerate() {
        let s = freq_to_sample(f, t);
        gz.push(s);
    }
    let wo = warmup_freqs.len();
    let mut first = None;
    let mut detected = false;
    for (t, &f) in freqs.iter().enumerate() {
        let s = freq_to_sample(f, wo + t);
        if gz.detect(s) {
            detected = true;
            if first.is_none() {
                first = Some(t);
            }
        }
    }
    (detected, first)
}

fn run_iir_trial(iir: &mut IirBandpass, freqs: &[f32], warmup_freqs: &[f32]) -> TrialResult {
    iir.reset();
    for (t, &f) in warmup_freqs.iter().enumerate() {
        let s = freq_to_sample(f, t);
        iir.push(s);
    }
    let wo = warmup_freqs.len();
    let mut first = None;
    let mut detected = false;
    for (t, &f) in freqs.iter().enumerate() {
        let s = freq_to_sample(f, wo + t);
        if iir.detect(s) {
            detected = true;
            if first.is_none() {
                first = Some(t);
            }
        }
    }
    (detected, first)
}

fn run_cricket_trial(brain: &mut CricketBrain, freqs: &[f32], warmup_freqs: &[f32]) -> TrialResult {
    brain.reset();
    for &f in warmup_freqs {
        let _ = brain.step(f);
    }
    let mut first = None;
    let mut detected = false;
    for (t, &f) in freqs.iter().enumerate() {
        let out = brain.step(f);
        if out > 0.0 {
            detected = true;
            if first.is_none() {
                first = Some(t);
            }
        }
    }
    (detected, first)
}

// ---------------------------------------------------------------------------
// Statistics accumulator
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Stats {
    tp: usize,
    fp: usize,
    tn: usize,
    fnn: usize,
    /// Sum of first-detection steps across TP trials (for latency avg).
    latency_sum: usize,
    latency_count: usize,
}

impl Stats {
    fn tpr(&self) -> f32 {
        let denom = (self.tp + self.fnn).max(1) as f32;
        self.tp as f32 / denom
    }

    fn fpr(&self) -> f32 {
        let denom = (self.fp + self.tn).max(1) as f32;
        self.fp as f32 / denom
    }

    fn avg_latency(&self) -> f32 {
        if self.latency_count == 0 {
            f32::NAN
        } else {
            self.latency_sum as f32 / self.latency_count as f32
        }
    }
}

fn record(stats: &mut Stats, result: TrialResult, target_present: bool) {
    let (detected, first_step) = result;
    match (target_present, detected) {
        (true, true) => {
            stats.tp += 1;
            if let Some(s) = first_step {
                stats.latency_sum += s;
                stats.latency_count += 1;
            }
        }
        (true, false) => stats.fnn += 1,
        (false, true) => stats.fp += 1,
        (false, false) => stats.tn += 1,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    const SEED: u64 = 1337;
    const TRIALS_PER_CLASS: usize = 120;

    let snr_levels: Vec<i32> = (-10..=30).step_by(5).collect();

    // Detector thresholds — tuned to give reasonable operating points.
    // These are fixed (not swept) so the comparison is fair: one working-point
    // per method, chosen conservatively to balance TPR and FPR.
    let mf_threshold: f32 = 0.25;
    let gz_threshold: f32 = 0.30;
    let iir_threshold: f32 = 0.05;

    // Goertzel block size: 30 samples ≈ half the signal window.
    let gz_block: usize = 30;

    let mut mf = MatchedFilter::new(mf_threshold);
    let mut gz = GoertzelDetector::new(gz_block, gz_threshold);
    let mut iir = IirBandpass::new(iir_threshold);
    let mut brain = CricketBrain::new(BrainConfig::default()).expect("valid brain config");

    // Per-method, per-SNR stats.
    let n_snr = snr_levels.len();
    let mut mf_stats: Vec<Stats> = (0..n_snr).map(|_| Stats::default()).collect();
    let mut gz_stats: Vec<Stats> = (0..n_snr).map(|_| Stats::default()).collect();
    let mut iir_stats: Vec<Stats> = (0..n_snr).map(|_| Stats::default()).collect();
    let mut cb_stats: Vec<Stats> = (0..n_snr).map(|_| Stats::default()).collect();

    for (si, &snr_db) in snr_levels.iter().enumerate() {
        // Each SNR level gets its own seeded RNG so results don't depend on
        // iteration order — matching the research_gen.rs pattern.
        let mut rng = Lcg::new(SEED ^ (snr_db as u64).wrapping_mul(0xDEAD_BEEF_1337));

        // ---- target_present trials ----
        for _ in 0..TRIALS_PER_CLASS {
            // build_trial_signal returns (warmup, observation) — single RNG source.
            let (warmup, obs) = build_trial_signal(&mut rng, snr_db, true);

            let r_mf = run_mf_trial(&mut mf, &obs, &warmup);
            let r_gz = run_goertzel_trial(&mut gz, &obs, &warmup);
            let r_iir = run_iir_trial(&mut iir, &obs, &warmup);
            let r_cb = run_cricket_trial(&mut brain, &obs, &warmup);

            record(&mut mf_stats[si], r_mf, true);
            record(&mut gz_stats[si], r_gz, true);
            record(&mut iir_stats[si], r_iir, true);
            record(&mut cb_stats[si], r_cb, true);
        }

        // ---- target_absent trials ----
        for _ in 0..TRIALS_PER_CLASS {
            let (warmup, obs) = build_trial_signal(&mut rng, snr_db, false);

            let r_mf = run_mf_trial(&mut mf, &obs, &warmup);
            let r_gz = run_goertzel_trial(&mut gz, &obs, &warmup);
            let r_iir = run_iir_trial(&mut iir, &obs, &warmup);
            let r_cb = run_cricket_trial(&mut brain, &obs, &warmup);

            record(&mut mf_stats[si], r_mf, false);
            record(&mut gz_stats[si], r_gz, false);
            record(&mut iir_stats[si], r_iir, false);
            record(&mut cb_stats[si], r_cb, false);
        }
    }

    // ---------------------------------------------------------------------------
    // Markdown output
    // ---------------------------------------------------------------------------

    println!("# Baseline Comparison — CricketBrain vs Classical Detectors\n");
    println!("Parameters: seed={SEED}, trials_per_class={TRIALS_PER_CLASS}, warmup=24");
    println!("Thresholds: MatchedFilter={mf_threshold:.2}, Goertzel={gz_threshold:.2}, IIR={iir_threshold:.2}\n");

    let header = "| SNR (dB) | Method          | TPR    | FPR    | Latency (steps) |";
    let sep = "|----------|-----------------|--------|--------|-----------------|";
    println!("{header}");
    println!("{sep}");

    let methods = [
        "MatchedFilter",
        "Goertzel     ",
        "IIR-Bandpass ",
        "CricketBrain ",
    ];
    let all_stats: [&Vec<Stats>; 4] = [&mf_stats, &gz_stats, &iir_stats, &cb_stats];

    for (si, &snr_db) in snr_levels.iter().enumerate() {
        for (mi, (method, stats)) in methods.iter().zip(all_stats.iter()).enumerate() {
            let s = &stats[si];
            let lat = s.avg_latency();
            let lat_str = if lat.is_nan() {
                "  N/A  ".to_string()
            } else {
                format!("{lat:7.1}")
            };
            // Print SNR only on first method row for readability.
            if mi == 0 {
                println!(
                    "| {:>8} | {method} | {:.4} | {:.4} | {lat_str:>15} |",
                    snr_db,
                    s.tpr(),
                    s.fpr(),
                );
            } else {
                println!(
                    "| {:>8} | {method} | {:.4} | {:.4} | {lat_str:>15} |",
                    "",
                    s.tpr(),
                    s.fpr(),
                );
            }
        }
        println!("{sep}");
    }

    // ---------------------------------------------------------------------------
    // CSV output
    // ---------------------------------------------------------------------------

    let out_dir = PathBuf::from("target/research");
    fs::create_dir_all(&out_dir).expect("create output dir");
    let csv_path = out_dir.join("baseline_comparison.csv");

    let mut csv = String::from("snr_db,method,tp,fp,tn,fn,tpr,fpr,avg_latency_steps\n");

    let method_names = ["MatchedFilter", "Goertzel", "IIR-Bandpass", "CricketBrain"];
    for (si, &snr_db) in snr_levels.iter().enumerate() {
        for (mi, &name) in method_names.iter().enumerate() {
            let s = &all_stats[mi][si];
            let lat = s.avg_latency();
            let lat_str = if lat.is_nan() {
                String::from("NaN")
            } else {
                format!("{lat:.3}")
            };
            csv.push_str(&format!(
                "{},{},{},{},{},{},{:.6},{:.6},{}\n",
                snr_db,
                name,
                s.tp,
                s.fp,
                s.tn,
                s.fnn,
                s.tpr(),
                s.fpr(),
                lat_str,
            ));
        }
    }

    fs::write(&csv_path, &csv).expect("write CSV");
    println!("\nCSV written to: {}", csv_path.display());
}
