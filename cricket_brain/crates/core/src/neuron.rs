// SPDX-License-Identifier: AGPL-3.0-only
use alloc::collections::VecDeque;
use core::mem;

use crate::memory::{MemoryStats, EMBEDDED_RAM_LIMIT_BYTES, SAMPLE_BYTES};

/// Default delay taps for embedded sizing checks.
pub const DEFAULT_NEURON_DELAY_TAPS: usize = 4;
const DEFAULT_NEURON_HISTORY_BYTES: usize = (DEFAULT_NEURON_DELAY_TAPS + 1) * SAMPLE_BYTES;
const _: [(); EMBEDDED_RAM_LIMIT_BYTES - DEFAULT_NEURON_HISTORY_BYTES] =
    [(); EMBEDDED_RAM_LIMIT_BYTES - DEFAULT_NEURON_HISTORY_BYTES];

/// Runtime configuration for a [`Neuron`].
#[derive(Debug, Clone, Copy)]
pub struct NeuronConfig {
    /// Hard squelch floor in normalized resonance space.
    ///
    /// Resonance responses below this floor are treated as noise.
    pub min_activation_threshold: f32,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            min_activation_threshold: 0.0,
        }
    }
}

/// A biomorphic neuron modeled after the auditory interneurons in the cricket
/// nervous system (Münster model). Each neuron has a characteristic eigenfrequency
/// and acts as a resonator with phase-locking capability.
///
/// The neuron uses Gaussian tuning for frequency selectivity and maintains a
/// history buffer for coincidence detection across delay lines.
///
/// Frequency selectivity is modeled as a Gaussian resonance curve:
///
/// \[
/// R(f, f_0, \sigma) = e^{-\frac{(f-f_0)^2}{2\sigma^2}}
/// \]
///
/// where:
/// - \(f\) is the input frequency,
/// - \(f_0\) is the neuron's eigenfrequency,
/// - \(\sigma\) is the effective bandwidth parameter.
///
/// Adaptive thresholding/squelch can be interpreted as a simple CFAR-style
/// gate on resonance:
///
/// \[
/// \hat{R}(t) = \begin{cases}
/// 0, & R(t) < \theta_{\min} \\
/// R(t), & R(t) \ge \theta_{\min}
/// \end{cases}
/// \]
///
/// where \(\theta_{\min}\) is `min_activation_threshold`.
#[derive(Debug, Clone)]
pub struct Neuron {
    /// Unique identifier for this neuron in the network.
    pub id: usize,
    /// Eigenfrequency (Hz) this neuron is tuned to.
    pub eigenfreq: f32,
    /// Current oscillation phase (0.0 to 1.0).
    pub phase: f32,
    /// Current activation amplitude (0.0 to 1.0).
    pub amplitude: f32,
    /// Firing threshold — neuron fires when amplitude exceeds this value.
    pub threshold: f32,
    /// Number of timesteps to look back for coincidence detection.
    pub delay_taps: usize,
    /// Ring buffer storing amplitude history for coincidence detection.
    /// Capacity = delay_taps + 1, so index 0 is the oldest sample.
    pub history: VecDeque<f32>,
    /// Minimum activation floor to suppress low-energy noise responses.
    pub min_activation_threshold: f32,
    /// Gaussian bandwidth parameter (default: 0.1 = 10% of eigenfrequency).
    /// Smaller values yield sharper frequency selectivity.
    pub bandwidth: f32,
    /// Exponential moving average of amplitude for homeostatic plasticity.
    /// Updated each step: `ema = 0.99 * ema + 0.01 * amplitude`.
    pub activity_ema: f32,
    /// Timestep of the most recent spike (coincidence fire). 0 = never spiked.
    /// Used by STDP to compute pre/post spike timing.
    pub last_spike_step: u32,
}

impl Neuron {
    /// Creates a new neuron tuned to the given frequency.
    ///
    /// # Arguments
    /// * `id` - Unique neuron identifier
    /// * `freq` - Eigenfrequency in Hz (e.g., 4500.0 for cricket carrier)
    /// * `delay_ms` - Delay tap length in milliseconds for coincidence detection
    ///
    /// # Example
    /// ```
    /// use cricket_brain_core::neuron::Neuron;
    /// let an1 = Neuron::new(0, 4500.0, 4);
    /// assert_eq!(an1.eigenfreq, 4500.0);
    /// ```
    pub fn new(id: usize, freq: f32, delay_ms: usize) -> Self {
        Self::new_with_config(id, freq, delay_ms, NeuronConfig::default())
    }

    /// Creates a new neuron tuned to the given frequency with custom config.
    pub fn new_with_config(id: usize, freq: f32, delay_ms: usize, config: NeuronConfig) -> Self {
        debug_assert!(
            freq.is_finite() && freq > 0.0,
            "eigenfrequency must be positive"
        );
        debug_assert!(
            config.min_activation_threshold.is_finite()
                && (0.0..=1.0).contains(&config.min_activation_threshold),
            "min_activation_threshold must be in [0, 1]"
        );
        let capacity = delay_ms + 1;
        let mut history = VecDeque::with_capacity(capacity);
        for _ in 0..capacity {
            history.push_back(0.0);
        }
        Neuron {
            id,
            eigenfreq: freq,
            phase: 0.0,
            amplitude: 0.0,
            threshold: 0.7,
            delay_taps: delay_ms,
            history,
            min_activation_threshold: config.min_activation_threshold,
            bandwidth: 0.1,
            activity_ema: 0.0,
            last_spike_step: 0,
        }
    }

    /// Resonance function using Gaussian frequency tuning.
    ///
    /// Computes match strength between the input frequency and the neuron's
    /// eigenfrequency using a Gaussian filter:
    ///
    /// ```text
    /// match = exp( -(Δf / f₀ / w)² )
    /// ```
    ///
    /// where `Δf = |input_freq - eigenfreq|`, `f₀ = eigenfreq`, and `w = 0.1`
    /// (10% bandwidth).
    ///
    /// If `match > 0.3`:
    /// - Amplitude increases: `A(t+1) = min(A(t) + match * 0.3, 1.0)`
    /// - Phase locks: `φ(t+1) = φ(t) + (φ_in - φ(t)) * 0.1`
    ///
    /// If `match <= 0.3`:
    /// - Amplitude decays: `A(t+1) = A(t) * 0.95`
    /// - Phase drifts toward zero: `φ(t+1) = φ(t) * 0.98`
    ///
    /// # Arguments
    /// * `input_freq` - The incoming signal frequency in Hz
    /// * `input_phase` - The incoming signal phase (0.0 to 1.0)
    ///
    /// # Returns
    /// The current amplitude after update.
    #[inline(always)]
    pub fn resonate(&mut self, input_freq: f32, input_phase: f32) -> f32 {
        debug_assert!(
            self.eigenfreq.is_finite() && self.eigenfreq > 0.0,
            "eigenfrequency must be positive",
        );
        debug_assert!(
            input_freq.is_finite() && input_freq > 0.0,
            "input frequency must be positive"
        );
        // Gaussian tuning curve: match = exp(-(Δf / f₀ / w)²)
        let delta_f = (input_freq - self.eigenfreq).abs();
        debug_assert!(self.bandwidth > 0.0, "gaussian width must be non-zero");
        let normalized = delta_f / self.eigenfreq / self.bandwidth;
        let match_strength = libm::expf(-normalized * normalized);

        let effective_match = if match_strength < self.min_activation_threshold {
            0.0
        } else {
            match_strength
        };

        if effective_match > 0.3 {
            // Resonance: amplitude grows, phase locks
            // A(t+1) = min(A(t) + match * 0.3, 1.0)
            self.amplitude = (self.amplitude + effective_match * 0.3).min(1.0);
            // φ(t+1) = φ(t) + (φ_in - φ(t)) * 0.1
            self.phase += (input_phase - self.phase) * 0.1;
        } else {
            // No resonance: decay
            // A(t+1) = A(t) * 0.95
            self.amplitude *= 0.95;
            // BUG #1 FIX: Phase also decays when not resonating
            // φ(t+1) = φ(t) * 0.98
            self.phase *= 0.98;
        }

        // Update history ring buffer for coincidence detection
        if self.history.len() == self.delay_taps + 1 {
            self.history.pop_front();
        }
        self.history.push_back(self.amplitude);

        // Update activity EMA for homeostatic plasticity
        self.activity_ema = 0.99 * self.activity_ema + 0.01 * self.amplitude;

        self.amplitude
    }

    /// Checks if this neuron should fire based on coincidence detection.
    ///
    /// A neuron fires when both the current amplitude AND the delayed amplitude
    /// (from `delay_taps` timesteps ago) exceed their respective thresholds:
    ///
    /// ```text
    /// fire = (A(t) > θ) ∧ (A(t - τ) > θ * 0.8)
    /// ```
    ///
    /// where `θ` is the firing threshold and `τ` is the delay tap length.
    ///
    /// # Returns
    /// `true` if the coincidence condition is met.
    pub fn check_coincidence(&self) -> bool {
        // BUG #4 FIX: Read index 0 (oldest element) instead of index delay_taps
        // (which would be the newest/current element).
        // self.history[0] is the amplitude from delay_taps timesteps ago.
        let delayed = self.history[0];
        self.amplitude > self.threshold && delayed > self.threshold * 0.8
    }

    /// Passive decay applied each timestep when no input is present.
    ///
    /// ```text
    /// A(t+1) = A(t) * 0.95
    /// φ(t+1) = φ(t) * 0.98
    /// ```
    #[inline(always)]
    pub fn decay(&mut self) {
        self.amplitude *= 0.95;
        self.phase *= 0.98;
    }

    /// Returns the neuron's current resonance/amplitude level.
    #[inline]
    pub fn resonance_level(&self) -> f32 {
        self.amplitude
    }

    /// Returns the neuron's current oscillation phase.
    #[inline]
    pub fn current_phase(&self) -> f32 {
        self.phase
    }

    /// Returns a read-only view of the amplitude history buffer.
    #[inline]
    pub fn history_buffer(&self) -> &VecDeque<f32> {
        &self.history
    }

    /// Returns the configured minimum activation floor.
    #[inline]
    pub fn min_activation_threshold(&self) -> f32 {
        self.min_activation_threshold
    }

    /// Estimates memory requirements for this neuron.
    ///
    /// - `static_bytes`: `size_of::<Neuron>()`
    /// - `dynamic_bytes`: history length × `size_of::<f32>()`
    #[inline]
    pub fn calculate_memory_requirements(&self) -> MemoryStats {
        MemoryStats {
            static_bytes: mem::size_of::<Self>(),
            dynamic_bytes: self.history.len() * mem::size_of::<f32>(),
        }
    }
}
