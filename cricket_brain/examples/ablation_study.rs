// SPDX-License-Identifier: AGPL-3.0-only
//! Systematic ablation study for the CricketBrain Münster circuit.
//!
//! Tests 6 configurations across SNR levels to measure the contribution of
//! each architectural component (LN2, LN3, LN5 interneurons, coincidence
//! detection, and delay lines) to detection performance.
//!
//! Outputs:
//!   - Markdown table to stdout
//!   - CSV to target/research/ablation_study.csv
//!
//! Usage:
//! ```bash
//! cargo run --release --example ablation_study
//! ```

use cricket_brain::brain::{BrainConfig, CricketBrain};
use cricket_brain::synapse::DelaySynapse;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Deterministic LCG (copied from research_gen.rs)
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
// Signal generators (identical to research_gen.rs)
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
// Ablation variant descriptors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AblationKind {
    /// Full circuit — no modifications.
    Full,
    /// Zero out the AN1→LN2 (idx 0) and LN2→ON1 (idx 3) synapse ring buffers
    /// and permanently suppress those paths by keeping them zeroed after reset.
    WithoutLn2,
    /// Zero out AN1→LN3 (idx 1) and LN3→ON1 (idx 4).
    WithoutLn3,
    /// Zero out AN1→LN5 (idx 2) and LN5→ON1 (idx 5).
    WithoutLn5,
    /// Use only instantaneous amplitude threshold — skip the delayed coincidence
    /// check that `CricketBrain::step()` performs internally. We achieve this by
    /// zeroing out ON1's history buffer every step before `step()` runs so the
    /// `delayed > dynamic_threshold * 0.8` gate always passes when amplitude is
    /// non-zero. Actually the simpler approach: intercept output by comparing
    /// ON1 amplitude directly after calling step; but since step() already gates,
    /// we instead build a thin wrapper that zeroes out ON1's history before each
    /// step so the delayed check never blocks.
    WithoutCoincidence,
    /// Replace all synapse delays with 1 (minimum possible).
    WithoutDelayLines,
}

impl AblationKind {
    fn name(self) -> &'static str {
        match self {
            AblationKind::Full => "Full circuit (baseline)",
            AblationKind::WithoutLn2 => "Without LN2 (inh, 3 ms)",
            AblationKind::WithoutLn3 => "Without LN3 (exc, 2 ms)",
            AblationKind::WithoutLn5 => "Without LN5 (inh, 5 ms)",
            AblationKind::WithoutCoincidence => "Without coincidence detection",
            AblationKind::WithoutDelayLines => "Without delay lines (all d=1)",
        }
    }

    fn short_name(self) -> &'static str {
        match self {
            AblationKind::Full => "full",
            AblationKind::WithoutLn2 => "no_ln2",
            AblationKind::WithoutLn3 => "no_ln3",
            AblationKind::WithoutLn5 => "no_ln5",
            AblationKind::WithoutCoincidence => "no_coincidence",
            AblationKind::WithoutDelayLines => "no_delay_lines",
        }
    }
}

// ---------------------------------------------------------------------------
// Brain factory — builds a brain then applies structural modifications
// ---------------------------------------------------------------------------

/// Synapse layout for the canonical 5-neuron circuit:
///   idx 0: AN1(0) → LN2(1), delay=3, inhibitory
///   idx 1: AN1(0) → LN3(2), delay=2, excitatory
///   idx 2: AN1(0) → LN5(3), delay=5, inhibitory
///   idx 3: LN2(1) → ON1(4), delay=1, inhibitory
///   idx 4: LN3(2) → ON1(4), delay=1, excitatory
///   idx 5: LN5(3) → ON1(4), delay=1, inhibitory
const AN1_TO_LN2: usize = 0;
const AN1_TO_LN3: usize = 1;
const AN1_TO_LN5: usize = 2;
const LN2_TO_ON1: usize = 3;
const LN3_TO_ON1: usize = 4;
const LN5_TO_ON1: usize = 5;

fn build_brain(kind: AblationKind, seed: u64) -> CricketBrain {
    let cfg = BrainConfig::default().with_seed(seed);

    match kind {
        AblationKind::Full => CricketBrain::new(cfg).expect("valid brain config"),

        AblationKind::WithoutLn2 => {
            let mut brain = CricketBrain::new(cfg).expect("valid brain config");
            // Replace AN1→LN2 and LN2→ON1 with zero-weight stubs.
            // We keep the synapses in place (same indices) but replace them
            // with a 1-step delay that always outputs 0.  The simplest way:
            // replace with a non-inhibitory, non-excitatory synapse that
            // connects to a dummy target (self-loop on AN1 is safest, but
            // could confuse things). Better: keep same topology but set both
            // synapses to target a scratch neuron that has no downstream path.
            //
            // Cleanest: just zero the ring buffers AND keep them zeroed by
            // overriding after each reset.  We achieve permanent suppression
            // by completely replacing those two synapses with ones that have
            // `from == to` on a neuron with no downstream significance (AN1
            // self-loop) so they never inject into ON1.
            brain.synapses[AN1_TO_LN2] = DelaySynapse::new(0, 0, 1, false); // AN1 self-loop (noop)
            brain.synapses[LN2_TO_ON1] = DelaySynapse::new(0, 0, 1, false); // AN1 self-loop (noop)
            brain
        }

        AblationKind::WithoutLn3 => {
            let mut brain = CricketBrain::new(cfg).expect("valid brain config");
            brain.synapses[AN1_TO_LN3] = DelaySynapse::new(0, 0, 1, false);
            brain.synapses[LN3_TO_ON1] = DelaySynapse::new(0, 0, 1, false);
            brain
        }

        AblationKind::WithoutLn5 => {
            let mut brain = CricketBrain::new(cfg).expect("valid brain config");
            brain.synapses[AN1_TO_LN5] = DelaySynapse::new(0, 0, 1, false);
            brain.synapses[LN5_TO_ON1] = DelaySynapse::new(0, 0, 1, false);
            brain
        }

        AblationKind::WithoutCoincidence => {
            // Standard brain; we'll zero ON1's history before every step in
            // run_trial_ablated so the coincidence gate always passes.
            CricketBrain::new(cfg).expect("valid brain config")
        }

        AblationKind::WithoutDelayLines => {
            let mut brain = CricketBrain::new(cfg).expect("valid brain config");
            // Rebuild every synapse with delay=1, preserving from/to/inhibitory.
            let rebuilt: Vec<DelaySynapse> = brain
                .synapses
                .iter()
                .map(|s| DelaySynapse::new(s.from, s.to, 1, s.inhibitory))
                .collect();
            brain.synapses = rebuilt;
            brain
        }
    }
}

// ---------------------------------------------------------------------------
// Trial runner
// ---------------------------------------------------------------------------

/// Run a single detection trial.
///
/// For `WithoutCoincidence` we zero ON1's history buffer before each step so
/// the delayed coincidence check inside `CricketBrain::step()` never blocks a
/// genuine amplitude spike — effectively disabling the delay requirement.
fn run_trial(
    brain: &mut CricketBrain,
    rng: &mut Lcg,
    snr_db: i32,
    target_present: bool,
    kind: AblationKind,
) -> bool {
    brain.reset();

    // Warm-up: 24 steps of background
    for _ in 0..24 {
        if kind == AblationKind::WithoutCoincidence {
            zero_on1_history(brain);
        }
        let _ = brain.step(background_freq(rng, snr_db));
    }

    // Observation window: 120 steps, signal in [32, 92)
    let mut detected = false;
    for t in 0..120 {
        if kind == AblationKind::WithoutCoincidence {
            zero_on1_history(brain);
        }
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

/// Zero ON1's history buffer so the delayed-coincidence gate inside step() is
/// bypassed (the oldest entry is 0.0, which passes `delayed > threshold * 0.8`
/// only when threshold is also 0 — so we instead set history to match the
/// current amplitude by pre-filling it with the threshold value to force pass).
///
/// More precisely: `step()` checks `delayed > dynamic_threshold * 0.8`.
/// We pre-fill ON1's history with a large value (1.0) so the historical check
/// always passes, leaving detection gated only on `amplitude > dynamic_threshold`.
#[inline]
fn zero_on1_history(brain: &mut CricketBrain) {
    let on1 = &mut brain.neurons[4]; // ON1 is always index 4
    for v in on1.history.iter_mut() {
        *v = 1.0; // Force historical amplitude to max — coincidence always "passes"
    }
}

// ---------------------------------------------------------------------------
// Study execution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct AblationResult {
    kind: AblationKind,
    snr_db: i32,
    tp: usize,
    fp: usize,
    tn: usize,
    fnn: usize,
}

impl AblationResult {
    fn tpr(&self) -> f32 {
        let denom = (self.tp + self.fnn).max(1) as f32;
        self.tp as f32 / denom
    }

    fn fpr(&self) -> f32 {
        let denom = (self.fp + self.tn).max(1) as f32;
        self.fp as f32 / denom
    }
}

fn run_ablation_study() -> Vec<AblationResult> {
    const SEED: u64 = 1337;
    const TRIALS_PER_CLASS: usize = 200;
    let snr_levels: &[i32] = &[-10, -5, 0, 5, 10, 15, 20];

    let kinds = [
        AblationKind::Full,
        AblationKind::WithoutLn2,
        AblationKind::WithoutLn3,
        AblationKind::WithoutLn5,
        AblationKind::WithoutCoincidence,
        AblationKind::WithoutDelayLines,
    ];

    let mut results = Vec::new();

    for &kind in &kinds {
        eprintln!("Running ablation: {}", kind.name());

        // Unique seed per ablation variant to match style of research_gen.rs
        let variant_seed = SEED ^ (kind.short_name().len() as u64 * 0xDEAD_BEEF);

        let mut brain = build_brain(kind, variant_seed);
        let mut rng = Lcg::new(variant_seed);

        for &snr_db in snr_levels {
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut tn = 0usize;
            let mut fnn = 0usize;

            // Signal-present trials
            for _ in 0..TRIALS_PER_CLASS {
                if run_trial(&mut brain, &mut rng, snr_db, true, kind) {
                    tp += 1;
                } else {
                    fnn += 1;
                }
            }

            // Signal-absent trials
            for _ in 0..TRIALS_PER_CLASS {
                if run_trial(&mut brain, &mut rng, snr_db, false, kind) {
                    fp += 1;
                } else {
                    tn += 1;
                }
            }

            results.push(AblationResult {
                kind,
                snr_db,
                tp,
                fp,
                tn,
                fnn,
            });
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn print_markdown_table(results: &[AblationResult]) {
    let snr_display = [0i32, 10, 20];

    println!();
    println!("## CricketBrain Ablation Study Results\n");

    // Header
    print!("| Configuration");
    for &snr in &snr_display {
        print!(" | SNR {:+}dB TPR | SNR {:+}dB FPR", snr, snr);
    }
    println!(" |");

    // Separator
    print!("|---");
    for _ in &snr_display {
        print!("|---:|---:");
    }
    println!("|");

    // One row per ablation kind
    let kinds = [
        AblationKind::Full,
        AblationKind::WithoutLn2,
        AblationKind::WithoutLn3,
        AblationKind::WithoutLn5,
        AblationKind::WithoutCoincidence,
        AblationKind::WithoutDelayLines,
    ];

    for kind in kinds {
        print!("| {}", kind.name());
        for &snr in &snr_display {
            let r = results
                .iter()
                .find(|r| r.kind == kind && r.snr_db == snr)
                .expect("result must exist");
            print!(" | {:.3} | {:.3}", r.tpr(), r.fpr());
        }
        println!(" |");
    }
    println!();
}

fn write_csv(results: &[AblationResult], output_dir: &Path) {
    let mut csv = String::from("configuration,short_name,snr_db,tp,fp,tn,fn,tpr,fpr\n");
    for r in results {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{:.6},{:.6}\n",
            r.kind.name(),
            r.kind.short_name(),
            r.snr_db,
            r.tp,
            r.fp,
            r.tn,
            r.fnn,
            r.tpr(),
            r.fpr(),
        ));
    }
    let path = output_dir.join("ablation_study.csv");
    fs::write(&path, csv).expect("write ablation CSV");
    eprintln!("CSV written to: {}", path.display());
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    eprintln!("CricketBrain Ablation Study");
    eprintln!("===========================");
    eprintln!("Configurations: 6");
    eprintln!("SNR levels: -10, -5, 0, 5, 10, 15, 20 dB");
    eprintln!("Trials per class per SNR: 200");
    eprintln!();

    let results = run_ablation_study();

    // Print markdown table to stdout
    print_markdown_table(&results);

    // Write CSV
    let output_dir = PathBuf::from("target/research");
    fs::create_dir_all(&output_dir).expect("create output dir");
    write_csv(&results, &output_dir);

    eprintln!("Done.");
}
