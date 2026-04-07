// SPDX-License-Identifier: AGPL-3.0-only
#![cfg_attr(not(feature = "std"), no_std)]

//! # Cricket-Brain
//!
//! A biomorphic AI inference engine based on the Münster model of cricket hearing.
//!
//! Cricket-Brain uses **delay-line coincidence detection** for pattern recognition.
//! No matrix multiplication, no CUDA, no weights — just biologically-inspired
//! resonators connected by delay synapses.
//!
//! ## Features
//!
//! - **v0.1**: Morse code recognition with 5-neuron cricket circuit
//! - **v0.2**: Multi-frequency token recognition via parallel resonator banks
//! - **v0.3**: Sequence prediction using delay-line pattern memory
//!
//! ## Quick Start
//!
//! ```rust
//! use cricket_brain::brain::CricketBrain;
//! use cricket_brain::patterns::{encode_morse, MORSE_FREQ};
//!
//! let mut brain = CricketBrain::new(Default::default()).unwrap();
//! let signal = encode_morse("SOS");
//!
//! for &(freq, duration) in &signal {
//!     for _ in 0..duration {
//!         let output = brain.step(freq);
//!         if output > 0.0 {
//!             println!("Spike! amplitude={output:.3}");
//!         }
//!     }
//! }
//! ```

extern crate alloc;

pub use cricket_brain_core::error;
pub use cricket_brain_core::logger;
pub use cricket_brain_core::memory;
pub use cricket_brain_core::neuron;
pub use cricket_brain_core::plasticity;
pub use cricket_brain_core::synapse;

pub mod brain;
#[cfg(feature = "cli")]
pub mod json_telemetry;
pub mod patterns;
pub mod resonator_bank;
pub mod sequence;
pub mod token;

/// Stable cross-language error-code contract (FFI/Python/WASM).
pub mod error_codes {
    pub const CRICKET_OK: i32 = 0;
    pub const CRICKET_ERR_NULL: i32 = 1;
    pub const CRICKET_ERR_INVALID_CONFIG: i32 = 2;
    pub const CRICKET_ERR_TOKEN_NOT_FOUND: i32 = 3;
    pub const CRICKET_ERR_INVALID_INPUT: i32 = 4;
    pub const CRICKET_ERR_INTERNAL: i32 = 255;
}

/// Ergonomic imports for building Sentinel-style applications.
pub mod prelude {
    pub use crate::brain::{BrainConfig, BrainMemorySummary, CricketBrain};
    pub use crate::error::CricketError;
    #[cfg(feature = "cli")]
    pub use crate::json_telemetry::JsonTelemetry;
    pub use crate::logger::{CricketLogger, NoopTelemetry, NullLogger, Telemetry};
    pub use crate::patterns::{decode_spikes, encode_morse, MORSE_FREQ};
    pub use crate::resonator_bank::{ResonatorBank, ResonatorChannel};
    pub use crate::sequence::{Prediction, PredictorConfig, SequencePredictor};
    pub use crate::token::{Token, TokenVocabulary};
}
