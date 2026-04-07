// SPDX-License-Identifier: AGPL-3.0-only
#![cfg(feature = "cli")]

use clap::Parser;
use cricket_brain::brain::{BrainConfig, BrainSnapshot, CricketBrain};
use cricket_brain::json_telemetry::JsonTelemetry;
use cricket_brain::logger::Telemetry;
use cricket_brain::sequence::{PredictorConfig, SequencePredictor};
use cricket_brain::token::TokenVocabulary;
use miette::{Context, IntoDiagnostic, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "cricket-cli", version, about = "Cricket Brain command center")]
struct Cli {
    /// Path to JSON/TOML config file.
    #[arg(long)]
    config: PathBuf,
    /// Input CSV file with one frequency per line or "timestamp,frequency".
    #[arg(long)]
    input: Option<PathBuf>,
    /// Run in live mode (sleep 1ms between samples).
    #[arg(long, default_value_t = false)]
    live: bool,
    /// Optional path to write telemetry JSONL; defaults to stdout.
    #[arg(long)]
    telemetry_out: Option<PathBuf>,
    /// Optional snapshot export path (JSON).
    #[arg(long)]
    snapshot_out: Option<PathBuf>,
    /// Optional snapshot restore path (JSON).
    #[arg(long)]
    resume_snapshot: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SentinelConfig {
    brain: BrainConfig,
    predictor: PredictorConfig,
    labels: Vec<String>,
    patterns: Vec<PatternConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternConfig {
    name: String,
    labels: Vec<String>,
}

fn parse_config(path: &Path) -> Result<SentinelConfig> {
    let raw = std::fs::read_to_string(path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read config file: {}", path.display()))?;
    match path.extension().and_then(|s| s.to_str()) {
        Some("json") => serde_json::from_str(&raw)
            .into_diagnostic()
            .wrap_err("invalid JSON config; expected keys: brain, predictor, labels, patterns"),
        Some("toml") => toml::from_str(&raw)
            .into_diagnostic()
            .wrap_err("invalid TOML config; expected keys: brain, predictor, labels, patterns"),
        _ => Err(miette::miette!(
            "unsupported config format. Use .json or .toml"
        )),
    }
}

fn read_signal_csv(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not open input CSV: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.into_diagnostic()?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let freq_str = line
            .split(',')
            .next_back()
            .ok_or_else(|| miette::miette!("invalid CSV row at line {}", line_no + 1))?;
        let freq: f32 = freq_str
            .trim()
            .parse()
            .into_diagnostic()
            .wrap_err_with(|| format!("invalid frequency at line {}", line_no + 1))?;
        if !freq.is_finite() || freq < 0.0 {
            return Err(miette::miette!(
                "invalid signal value at line {}: frequency must be finite and >= 0",
                line_no + 1
            ));
        }
        out.push(freq);
    }
    Ok(out)
}

fn build_predictor(config: &SentinelConfig) -> Result<SequencePredictor> {
    let label_refs: Vec<&str> = config.labels.iter().map(String::as_str).collect();
    let vocab = TokenVocabulary::from_labels(&label_refs);
    let mut predictor = SequencePredictor::new(vocab, config.predictor.clone())
        .map_err(|e| miette::miette!("failed to initialize sequence predictor from config: {e}"))?;

    for p in &config.patterns {
        let refs: Vec<&str> = p.labels.iter().map(String::as_str).collect();
        predictor
            .register_pattern(&p.name, &refs)
            .map_err(|e| miette::miette!("invalid pattern '{}': {e}", p.name))?;
    }
    Ok(predictor)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = parse_config(&cli.config)?;

    let mut brain = if let Some(snapshot_path) = &cli.resume_snapshot {
        let raw = std::fs::read_to_string(snapshot_path)
            .into_diagnostic()
            .wrap_err_with(|| format!("could not read snapshot: {}", snapshot_path.display()))?;
        let snapshot: BrainSnapshot = serde_json::from_str(&raw)
            .into_diagnostic()
            .wrap_err("snapshot JSON is malformed; expected BrainSnapshot structure")?;
        CricketBrain::from_snapshot(&snapshot)
            .map_err(|e| miette::miette!("failed to restore brain from snapshot: {e}"))?
    } else {
        CricketBrain::new(config.brain.clone())
            .map_err(|e| miette::miette!("failed to initialize brain from config: {e}"))?
    };

    let mut predictor = build_predictor(&config)?;

    let mut telemetry: Box<dyn Telemetry> = if let Some(path) = cli.telemetry_out.as_ref() {
        let file = File::create(path).into_diagnostic().wrap_err_with(|| {
            format!("could not create telemetry output file: {}", path.display())
        })?;
        Box::new(JsonTelemetry::new(file))
    } else {
        Box::new(JsonTelemetry::stdout())
    };

    let input = if let Some(path) = cli.input.as_ref() {
        read_signal_csv(path)?
    } else {
        vec![4_500.0; 500]
    };

    let mut signal_energy = 0.0f32;
    let mut noise_energy = 0.0f32;
    for (i, freq) in input.iter().copied().enumerate() {
        let out = brain.step_with_telemetry(freq, telemetry.as_mut());
        let _ = predictor.step(freq);
        if out > 0.0 {
            signal_energy += out * out;
        } else {
            noise_energy += 1e-6;
        }

        if let Some(pred) = predictor.predict() {
            telemetry.on_sequence_matched(
                pred.token_id,
                pred.confidence,
                pred.snr,
                pred.jitter,
                pred.tolerance,
            );
            eprintln!(
                "[match] pattern={} next={} confidence={:.3}",
                pred.pattern_name, pred.label, pred.confidence
            );
        }

        if cli.live {
            thread::sleep(Duration::from_millis(1));
        }

        if i % 1000 == 0 {
            let snr = (signal_energy + 1e-6) / (noise_energy + 1e-6);
            telemetry.on_snr_report(snr);
        }
    }

    if let Some(path) = cli.snapshot_out.as_ref() {
        let snapshot = brain.snapshot();
        let mut file = File::create(path)
            .into_diagnostic()
            .wrap_err_with(|| format!("could not create snapshot file: {}", path.display()))?;
        serde_json::to_writer_pretty(&mut file, &snapshot)
            .into_diagnostic()
            .wrap_err("failed to serialize snapshot to JSON")?;
        file.write_all(b"\n").into_diagnostic()?;
    }

    Ok(())
}
