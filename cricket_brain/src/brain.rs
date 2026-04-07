// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::CricketError;
use crate::neuron::Neuron;
use crate::synapse::DelaySynapse;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use cricket_brain_core::logger::Telemetry;
use cricket_brain_core::memory::MemoryStats;
use cricket_brain_core::neuron::NeuronConfig;
use cricket_brain_core::plasticity::{
    apply_homeostasis, apply_stdp, HomeostasisConfig, StdpConfig,
};

/// Configuration object for constructing a [`CricketBrain`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct BrainConfig {
    /// Number of neurons in the network.
    pub n_neurons: usize,
    /// Lower frequency bound used for scaled initialization.
    pub min_freq: f32,
    /// Upper frequency bound used for scaled initialization.
    pub max_freq: f32,
    /// Optional number of synaptic connections. If `None`, defaults to `n_neurons * 3`.
    pub k_connections: Option<usize>,
    /// Sample rate in Hz (default: 1000 = 1ms timesteps).
    pub sample_rate_hz: u32,
    /// Hard floor for neuron activation to suppress low-energy noise.
    pub min_activation_threshold: f32,
    /// Enable adaptive gain control for noisy environments.
    pub adaptive_sensitivity: bool,
    /// Exponential moving average rate used by AGC (0, 1].
    pub agc_rate: f32,
    /// Deterministic seed for internal phase/noise synthesis.
    pub seed: u64,
    /// Privacy-preserving telemetry mode (HIPAA/GDPR-friendly).
    pub privacy_mode: bool,
}

impl BrainConfig {
    /// Canonical 5-neuron cricket circuit configuration.
    pub fn standard() -> Self {
        Self::default()
    }

    /// Convenience constructor for scaled topologies.
    pub fn scaled(n_neurons: usize, k_connections: usize) -> Self {
        Self {
            n_neurons,
            min_freq: 2000.0,
            max_freq: 8000.0,
            k_connections: Some(k_connections),
            sample_rate_hz: 1000,
            min_activation_threshold: 0.0,
            adaptive_sensitivity: false,
            agc_rate: 0.01,
            seed: 0xC0DEC0DE5EEDu64,
            privacy_mode: false,
        }
    }

    /// Builder-style setter for neuron count.
    pub fn with_neurons(mut self, n_neurons: usize) -> Self {
        self.n_neurons = n_neurons;
        self
    }

    /// Builder-style setter for frequency range.
    pub fn with_freq_range(mut self, min_freq: f32, max_freq: f32) -> Self {
        self.min_freq = min_freq;
        self.max_freq = max_freq;
        self
    }

    /// Builder-style setter for synaptic connection count.
    pub fn with_connections(mut self, k_connections: usize) -> Self {
        self.k_connections = Some(k_connections);
        self
    }

    /// Builder-style setter for minimum activation threshold.
    pub fn with_min_activation_threshold(mut self, threshold: f32) -> Self {
        self.min_activation_threshold = threshold;
        self
    }

    /// Builder-style setter for adaptive sensitivity mode.
    pub fn with_adaptive_sensitivity(mut self, enabled: bool) -> Self {
        self.adaptive_sensitivity = enabled;
        self
    }

    /// Builder-style setter for deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Builder-style setter for privacy-preserving telemetry.
    pub fn with_privacy_mode(mut self, enabled: bool) -> Self {
        self.privacy_mode = enabled;
        self
    }

    /// Validate configuration before construction.
    pub fn validate(&self) -> Result<(), CricketError> {
        if self.n_neurons == 0 {
            return Err(CricketError::InvalidConfiguration(String::from(
                "n_neurons must be greater than 0",
            )));
        }
        if self.min_freq >= self.max_freq {
            return Err(CricketError::InvalidConfiguration(String::from(
                "min_freq must be lower than max_freq",
            )));
        }
        if self.sample_rate_hz == 0 {
            return Err(CricketError::InvalidConfiguration(String::from(
                "sample_rate_hz must be greater than 0",
            )));
        }
        if !self.min_activation_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.min_activation_threshold)
        {
            return Err(CricketError::InvalidConfiguration(String::from(
                "min_activation_threshold must be in [0, 1]",
            )));
        }
        if !self.agc_rate.is_finite() || self.agc_rate <= 0.0 || self.agc_rate > 1.0 {
            return Err(CricketError::InvalidConfiguration(String::from(
                "agc_rate must be in (0, 1]",
            )));
        }
        Ok(())
    }
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            n_neurons: 5,
            min_freq: 4000.0,
            max_freq: 5000.0,
            k_connections: Some(6),
            sample_rate_hz: 1000,
            min_activation_threshold: 0.0,
            adaptive_sensitivity: false,
            agc_rate: 0.01,
            seed: 0xC0DEC0DE5EEDu64,
            privacy_mode: false,
        }
    }
}

/// The CricketBrain is a biomorphic inference engine implementing the Münster
/// model of cricket auditory processing.
///
/// Architecture (5-neuron standard configuration):
/// ```text
/// AN1 ──┬──▶ LN2 (inh, 3ms) ──▶ ON1
///       ├──▶ LN5 (inh, 5ms) ──▶ ON1
///       └──▶ LN3 (exc, 2ms) ──▶ ON1
/// ```
///
/// - **AN1** (id=0): Auditory receptor neuron, 4500 Hz
/// - **LN2** (id=1): Inhibitory local interneuron, 4500 Hz, 3ms delay
/// - **LN3** (id=2): Excitatory local interneuron, 4500 Hz, 2ms delay
/// - **LN5** (id=3): Inhibitory local interneuron, 4500 Hz, 5ms delay
/// - **ON1** (id=4): Output neuron, 4500 Hz, 4ms coincidence window
#[derive(Debug, Clone)]
pub struct CricketBrain {
    /// All neurons in the network.
    pub neurons: Vec<Neuron>,
    /// All synaptic connections.
    pub synapses: Vec<DelaySynapse>,
    /// Current simulation timestep.
    pub time_step: usize,
    /// Sample rate in Hz (default: 1000 = 1ms timesteps).
    pub sample_rate_hz: u32,
    /// Adaptive gain control toggle.
    pub adaptive_sensitivity: bool,
    /// Current global sensitivity multiplier.
    pub global_sensitivity: f32,
    /// Input-energy exponential moving average used by AGC.
    input_energy_ema: f32,
    /// AGC smoothing factor.
    agc_rate: f32,
    /// Rolling entropy window (quantized input bins).
    input_bin_window: VecDeque<u8>,
    /// Histogram counts for entropy estimation.
    input_bin_counts: [usize; 16],
    /// Deterministic RNG state for internal phase dither.
    rng_state: u64,
    /// Seed used to reinitialize deterministic RNG on reset.
    initial_seed: u64,
    /// Privacy-preserving telemetry toggle.
    privacy_mode: bool,
    /// Last telemetry step for relative timestamp deltas.
    last_telemetry_step: usize,
    /// Optional STDP configuration. When `Some`, plasticity is applied
    /// after each spike event, adjusting synaptic weights online.
    stdp_config: Option<StdpConfig>,
    /// Optional homeostasis configuration. When `Some`, neuron thresholds
    /// are slowly adjusted to maintain a target activity level.
    homeostasis_config: Option<HomeostasisConfig>,
}

#[derive(Debug, Clone, Copy)]
pub struct SystemHealth {
    pub entropy: f32,
    pub active_neurons: usize,
    pub total_neurons: usize,
}

/// Aggregate RAM estimate for a full brain instance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BrainMemorySummary {
    /// Aggregated neuron memory stats.
    pub neuron_stats: MemoryStats,
    /// Aggregated synapse memory stats.
    pub synapse_stats: MemoryStats,
}

/// Portable brain state snapshot for pause/resume workflows.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct BrainSnapshot {
    pub version_hash: String,
    pub checksum: u64,
    pub neurons: Vec<NeuronSnapshot>,
    pub synapses: Vec<SynapseSnapshot>,
    pub time_step: usize,
    pub sample_rate_hz: u32,
    pub adaptive_sensitivity: bool,
    pub global_sensitivity: f32,
    pub input_energy_ema: f32,
    pub agc_rate: f32,
    pub rng_state: u64,
    pub initial_seed: u64,
    pub privacy_mode: bool,
    pub last_telemetry_step: usize,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct NeuronSnapshot {
    pub id: usize,
    pub eigenfreq: f32,
    pub phase: f32,
    pub amplitude: f32,
    pub threshold: f32,
    pub delay_taps: usize,
    pub history: Vec<f32>,
    pub min_activation_threshold: f32,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct SynapseSnapshot {
    pub from: usize,
    pub to: usize,
    pub delay_ms: usize,
    pub inhibitory: bool,
    pub ring_buffer: Vec<f32>,
}

impl BrainMemorySummary {
    /// Total estimated RAM usage in bytes.
    #[inline]
    pub const fn total_ram_bytes(self) -> usize {
        self.neuron_stats.total_bytes() + self.synapse_stats.total_bytes()
    }
}

impl CricketBrain {
    const SNAPSHOT_VERSION_HASH: &'static str = "cricket-brain-snapshot-v2";
    /// Creates a cricket brain from a configuration object.
    ///
    /// This is the canonical Münster model configuration optimized for
    /// detecting ~4500 Hz carrier frequency with pulse-interval sensitivity.
    ///
    /// # Example
    /// ```
    /// use cricket_brain::brain::CricketBrain;
    /// let brain = CricketBrain::new(Default::default()).unwrap();
    /// assert_eq!(brain.neurons.len(), 5);
    /// assert_eq!(brain.synapses.len(), 6);
    /// ```
    pub fn new(config: BrainConfig) -> Result<Self, CricketError> {
        config.validate()?;
        let k_connections = config.k_connections.unwrap_or(config.n_neurons * 3);
        let neuron_cfg = NeuronConfig {
            min_activation_threshold: config.min_activation_threshold,
        };

        if config.n_neurons == 5 && k_connections == 6 {
            // Canonical Münster-like 5-neuron circuit.
            let carrier = (config.min_freq + config.max_freq) * 0.5;
            let neurons = vec![
                Neuron::new_with_config(0, carrier, 4, neuron_cfg), // AN1: auditory receptor
                Neuron::new_with_config(1, carrier, 3, neuron_cfg), // LN2: inhibitory interneuron
                Neuron::new_with_config(2, carrier, 2, neuron_cfg), // LN3: excitatory interneuron
                Neuron::new_with_config(3, carrier, 5, neuron_cfg), // LN5: inhibitory interneuron
                Neuron::new_with_config(4, carrier, 4, neuron_cfg), // ON1: output neuron
            ];

            let synapses = vec![
                DelaySynapse::new(0, 1, 3, true),
                DelaySynapse::new(0, 2, 2, false),
                DelaySynapse::new(0, 3, 5, true),
                DelaySynapse::new(1, 4, 1, true),
                DelaySynapse::new(2, 4, 1, false),
                DelaySynapse::new(3, 4, 1, true),
            ];

            return Ok(CricketBrain {
                neurons,
                synapses,
                time_step: 0,
                sample_rate_hz: config.sample_rate_hz,
                adaptive_sensitivity: config.adaptive_sensitivity,
                global_sensitivity: 1.0,
                input_energy_ema: 0.0,
                agc_rate: config.agc_rate,
                input_bin_window: VecDeque::new(),
                input_bin_counts: [0; 16],
                rng_state: config.seed,
                initial_seed: config.seed,
                privacy_mode: config.privacy_mode,
                last_telemetry_step: 0,
                stdp_config: None,
                homeostasis_config: None,
            });
        }

        let n_neurons = config.n_neurons;
        let neurons: Vec<Neuron> = (0..n_neurons)
            .map(|i| {
                let freq = config.min_freq
                    + (i as f32 / n_neurons as f32) * (config.max_freq - config.min_freq);
                let delay = 2 + (i % 8);
                Neuron::new_with_config(i, freq, delay, neuron_cfg)
            })
            .collect();

        let synapses: Vec<DelaySynapse> = (0..k_connections)
            .map(|i| {
                let from = i % n_neurons;
                let to = (i * 7 + 13) % n_neurons;
                let delay = 1 + (i % 10);
                let inhibitory = i % 3 == 0;
                DelaySynapse::new(from, to, delay, inhibitory)
            })
            .collect();

        Ok(CricketBrain {
            neurons,
            synapses,
            time_step: 0,
            sample_rate_hz: config.sample_rate_hz,
            adaptive_sensitivity: config.adaptive_sensitivity,
            global_sensitivity: 1.0,
            input_energy_ema: 0.0,
            agc_rate: config.agc_rate,
            input_bin_window: VecDeque::new(),
            input_bin_counts: [0; 16],
            rng_state: config.seed,
            initial_seed: config.seed,
            privacy_mode: config.privacy_mode,
            last_telemetry_step: 0,
            stdp_config: None,
            homeostasis_config: None,
        })
    }

    /// Creates a scaled brain with `n_neurons` neurons and `k_connections` synapses.
    ///
    /// Neurons are assigned frequencies spanning 2000–8000 Hz uniformly.
    /// Connections are created in a feed-forward ring topology with random-ish delays.
    ///
    /// # Arguments
    /// * `n_neurons` - Number of neurons (e.g., 40960)
    /// * `k_connections` - Number of synaptic connections
    ///
    /// # Example
    /// ```
    /// use cricket_brain::brain::CricketBrain;
    /// let big = CricketBrain::new_scaled(1000, 3000).unwrap();
    /// assert_eq!(big.neurons.len(), 1000);
    /// assert_eq!(big.synapses.len(), 3000);
    /// ```
    pub fn new_scaled(n_neurons: usize, k_connections: usize) -> Result<Self, CricketError> {
        Self::new(BrainConfig::scaled(n_neurons, k_connections))
    }

    /// Advances the brain by one timestep with the given input frequency.
    ///
    /// Processing pipeline:
    /// 1. AN1 (neuron 0) resonates with the raw input frequency
    /// 2. All synapses transmit delayed signals
    /// 3. Downstream neurons resonate with incoming synaptic signals
    /// 4. Return ON1 (output neuron) amplitude
    ///
    /// # Arguments
    /// * `input_freq` - Input signal frequency in Hz (0.0 = silence)
    ///
    /// # Returns
    /// The output neuron's amplitude (0.0 to 1.0).
    pub fn step(&mut self, input_freq: f32) -> f32 {
        let phase_dither = self.next_phase_dither();
        let input_phase = ((self.time_step as f32 * 0.01) + phase_dither) % 1.0;
        let n = self.neurons.len();
        let is_silence = input_freq <= 0.0;
        self.update_adaptive_sensitivity(input_freq);
        self.update_input_entropy(input_freq);

        // Step 1: AN1 resonates with input
        // BUG #2 FIX: Pass input_freq (the actual input) instead of eigenfreq
        if is_silence {
            // No signal: aggressive decay (silence = no acoustic energy)
            self.neurons[0].amplitude *= 0.5;
            self.neurons[0].phase *= 0.5;
            // Update AN1 history during silence
            let n0 = &mut self.neurons[0];
            if n0.history.len() == n0.delay_taps + 1 {
                n0.history.pop_front();
            }
            let amp = n0.amplitude;
            n0.history.push_back(amp);
        } else {
            self.neurons[0].resonate(input_freq, input_phase);
        }

        // Step 2: Propagate through synapses and collect signals per target
        let mut incoming_signals = vec![0.0_f32; n];

        for synapse in &mut self.synapses {
            let source_amp = self.neurons[synapse.from].amplitude;
            let transmitted = synapse.transmit(source_amp);
            incoming_signals[synapse.to] += transmitted;
        }

        // Step 3: Update downstream neurons (skip AN1 at index 0)
        // Using range loop because we mutate neurons[i] based on incoming_signals[i]
        #[allow(clippy::needless_range_loop)]
        for i in 1..n {
            if is_silence {
                // During silence: only decay, no excitation from residual signals
                // Aggressive silence decay for all downstream neurons
                self.neurons[i].amplitude *= 0.5;
                self.neurons[i].phase *= 0.5;
                let ni = &mut self.neurons[i];
                if ni.history.len() == ni.delay_taps + 1 {
                    ni.history.pop_front();
                }
                let amp = ni.amplitude;
                ni.history.push_back(amp);
            } else {
                let signal = incoming_signals[i];
                if signal.abs() > 0.01 {
                    // BUG #2 FIX: Pass input_freq to resonate, not eigenfreq
                    self.neurons[i].resonate(input_freq, input_phase);
                    // Apply synaptic modulation to amplitude
                    self.neurons[i].amplitude = (self.neurons[i].amplitude
                        + signal * 0.2 * self.global_sensitivity)
                        .clamp(0.0, 1.0);
                } else {
                    self.neurons[i].decay();
                }
            }
        }

        self.time_step += 1;
        let ts = self.time_step as u32;

        // Step 4: Track spikes for STDP — update last_spike_step for neurons
        // that exceed their threshold (any neuron can "spike", not just ON1).
        for neuron in &mut self.neurons {
            if neuron.amplitude > neuron.threshold {
                neuron.last_spike_step = ts;
            }
        }

        // Step 5: Apply STDP if enabled — adjust weights based on spike timing.
        if let Some(ref config) = self.stdp_config {
            let config = *config;
            for syn in &mut self.synapses {
                let pre_time = self.neurons[syn.from].last_spike_step;
                let post_time = self.neurons[syn.to].last_spike_step;
                apply_stdp(syn, pre_time, post_time, &config);
            }
        }

        // Step 6: Apply homeostasis if enabled — adjust thresholds to target activity.
        if let Some(ref config) = self.homeostasis_config {
            let config = *config;
            for neuron in &mut self.neurons {
                apply_homeostasis(neuron, &config);
            }
        }

        // Step 7: Return output neuron amplitude (ON1, last neuron)
        let output_idx = n - 1;
        let dynamic_threshold =
            self.neurons[output_idx].threshold / self.global_sensitivity.max(0.001);
        let delayed = self.neurons[output_idx].history[0];
        if self.neurons[output_idx].amplitude > dynamic_threshold
            && delayed > dynamic_threshold * 0.8
        {
            self.neurons[output_idx].amplitude
        } else {
            0.0
        }
    }

    #[inline]
    fn next_phase_dither(&mut self) -> f32 {
        // xorshift64*; deterministic and no_std friendly.
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let mixed = x.wrapping_mul(2685821657736338717);
        let unit = (mixed as f64) / (u64::MAX as f64);
        (unit as f32) * 0.001
    }

    /// Step with telemetry emission.
    pub fn step_with_telemetry<T: Telemetry + ?Sized>(
        &mut self,
        input_freq: f32,
        telemetry: &mut T,
    ) -> f32 {
        let out = self.step(input_freq);
        let resonance = self.neurons[0].resonance_level();
        if self.privacy_mode {
            // Coarsen values to avoid leaking detailed input-correlated traces.
            let quantized = (resonance * 10.0).round() / 10.0;
            telemetry.on_resonance_change(0, quantized);
        } else {
            telemetry.on_resonance_change(0, resonance);
        }
        if out > 0.0 {
            let ts = if self.privacy_mode {
                let delta = self.time_step.saturating_sub(self.last_telemetry_step);
                self.last_telemetry_step = self.time_step;
                delta as u64
            } else {
                self.time_step as u64
            };
            telemetry.on_spike(self.neurons.len() - 1, ts);
        }
        if let Some(health) = self.health_check() {
            telemetry.on_system_overload(
                health.entropy,
                health.active_neurons,
                health.total_neurons,
            );
        }
        out
    }

    /// Updates global sensitivity using a low-pass estimate of input energy.
    ///
    /// Adaptive rule (CFAR-inspired):
    /// \[
    /// E_t = (1-\alpha)E_{t-1} + \alpha\,u_t,\quad
    /// g_t = \mathrm{clamp}(1.2 - 0.4E_t,\ 0.6,\ 1.4)
    /// \]
    /// where \(u_t \in \{0,1\}\) is instantaneous input energy and
    /// \(\alpha\) is `agc_rate`.
    fn update_adaptive_sensitivity(&mut self, input_freq: f32) {
        if !self.adaptive_sensitivity {
            self.global_sensitivity = 1.0;
            return;
        }
        let energy = if input_freq > 0.0 { 1.0 } else { 0.0 };
        self.input_energy_ema =
            (1.0 - self.agc_rate) * self.input_energy_ema + self.agc_rate * energy;
        self.global_sensitivity = (1.2 - 0.4 * self.input_energy_ema).clamp(0.6, 1.4);
    }

    fn update_input_entropy(&mut self, input_freq: f32) {
        let normalized = (input_freq / 1000.0).clamp(0.0, 15.999);
        let bin = normalized as u8;
        self.input_bin_window.push_back(bin);
        self.input_bin_counts[bin as usize] += 1;

        const WINDOW: usize = 128;
        if self.input_bin_window.len() > WINDOW {
            if let Some(old) = self.input_bin_window.pop_front() {
                self.input_bin_counts[old as usize] =
                    self.input_bin_counts[old as usize].saturating_sub(1);
            }
        }
    }

    /// Detects chaotic input / overload state.
    pub fn health_check(&self) -> Option<SystemHealth> {
        let total = self.input_bin_window.len();
        if total < 32 {
            return None;
        }
        let mut entropy = 0.0f32;
        for &count in &self.input_bin_counts {
            if count == 0 {
                continue;
            }
            let p = count as f32 / total as f32;
            entropy -= p * p.log2();
        }
        let active_neurons = self
            .neurons
            .iter()
            .filter(|n| n.amplitude > n.threshold)
            .count();
        let simultaneous = active_neurons * 10 >= self.neurons.len() * 8;
        if entropy > 3.2 && simultaneous {
            Some(SystemHealth {
                entropy,
                active_neurons,
                total_neurons: self.neurons.len(),
            })
        } else {
            None
        }
    }

    /// Processes a batch of input frequencies and returns all output amplitudes.
    ///
    /// Enables online STDP (Spike-Timing Dependent Plasticity).
    ///
    /// When enabled, synaptic weights are adjusted after each spike based
    /// on the relative timing of pre- and post-synaptic activity.
    /// This allows the network to adapt to input patterns over time.
    ///
    /// # Arguments
    /// * `config` - STDP learning parameters (learning rate, time constant, bounds)
    pub fn enable_stdp(&mut self, config: StdpConfig) {
        self.stdp_config = Some(config);
    }

    /// Disables online STDP. Existing weights are preserved.
    pub fn disable_stdp(&mut self) {
        self.stdp_config = None;
    }

    /// Returns the current STDP configuration, if enabled.
    pub fn stdp_config(&self) -> Option<&StdpConfig> {
        self.stdp_config.as_ref()
    }

    /// Enables homeostatic threshold adaptation.
    ///
    /// Neuron thresholds slowly adjust to maintain a target activity level:
    /// overactive neurons become harder to fire, quiet neurons become easier.
    pub fn enable_homeostasis(&mut self, config: HomeostasisConfig) {
        self.homeostasis_config = Some(config);
    }

    /// Disables homeostatic adaptation. Current thresholds are preserved.
    pub fn disable_homeostasis(&mut self) {
        self.homeostasis_config = None;
    }

    /// Returns the current homeostasis configuration, if enabled.
    pub fn homeostasis_config(&self) -> Option<&HomeostasisConfig> {
        self.homeostasis_config.as_ref()
    }

    /// # Arguments
    /// * `inputs` - Slice of input frequencies, one per timestep
    ///
    /// # Returns
    /// A vector of output amplitudes, one per timestep.
    pub fn step_batch(&mut self, inputs: &[f32]) -> Vec<f32> {
        inputs.iter().map(|&freq| self.step(freq)).collect()
    }

    /// Resets the brain to its initial state.
    ///
    /// Clears all neuron amplitudes, phases, histories, and synapse buffers.
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.amplitude = 0.0;
            neuron.phase = 0.0;
            for val in neuron.history.iter_mut() {
                *val = 0.0;
            }
        }
        for synapse in &mut self.synapses {
            for val in synapse.ring_buffer.iter_mut() {
                *val = 0.0;
            }
        }
        self.time_step = 0;
        self.rng_state = self.initial_seed;
        self.last_telemetry_step = 0;
    }

    /// Returns the approximate memory usage of this brain in bytes.
    ///
    /// Accounts for neuron structs, history buffers, synapse structs, and ring buffers.
    pub fn memory_usage_bytes(&self) -> usize {
        self.total_ram_estimate().total_ram_bytes()
    }

    /// Exports a full internal state snapshot.
    pub fn snapshot(&self) -> BrainSnapshot {
        let neurons = self
            .neurons
            .iter()
            .map(|n| NeuronSnapshot {
                id: n.id,
                eigenfreq: n.eigenfreq,
                phase: n.phase,
                amplitude: n.amplitude,
                threshold: n.threshold,
                delay_taps: n.delay_taps,
                history: n.history.iter().copied().collect(),
                min_activation_threshold: n.min_activation_threshold,
            })
            .collect();
        let synapses = self
            .synapses
            .iter()
            .map(|s| SynapseSnapshot {
                from: s.from,
                to: s.to,
                delay_ms: s.delay_ms,
                inhibitory: s.inhibitory,
                ring_buffer: s.ring_buffer.iter().copied().collect(),
            })
            .collect();
        let mut snapshot = BrainSnapshot {
            version_hash: Self::SNAPSHOT_VERSION_HASH.to_string(),
            checksum: 0,
            neurons,
            synapses,
            time_step: self.time_step,
            sample_rate_hz: self.sample_rate_hz,
            adaptive_sensitivity: self.adaptive_sensitivity,
            global_sensitivity: self.global_sensitivity,
            input_energy_ema: self.input_energy_ema,
            agc_rate: self.agc_rate,
            rng_state: self.rng_state,
            initial_seed: self.initial_seed,
            privacy_mode: self.privacy_mode,
            last_telemetry_step: self.last_telemetry_step,
        };
        snapshot.checksum = Self::snapshot_checksum(&snapshot);
        snapshot
    }

    /// Restores this instance from a previously exported snapshot.
    pub fn restore_from_snapshot(&mut self, snapshot: &BrainSnapshot) -> Result<(), CricketError> {
        if snapshot.version_hash != Self::SNAPSHOT_VERSION_HASH {
            return Err(CricketError::InvalidInput(String::from(
                "snapshot version hash mismatch",
            )));
        }
        if Self::snapshot_checksum(snapshot) != snapshot.checksum {
            return Err(CricketError::InvalidInput(String::from(
                "snapshot checksum mismatch",
            )));
        }
        if snapshot.neurons.is_empty() {
            return Err(CricketError::InvalidInput(String::from(
                "snapshot must contain at least one neuron",
            )));
        }
        self.neurons = snapshot
            .neurons
            .iter()
            .map(|n| Neuron {
                id: n.id,
                eigenfreq: n.eigenfreq,
                phase: n.phase,
                amplitude: n.amplitude,
                threshold: n.threshold,
                delay_taps: n.delay_taps,
                history: VecDeque::from(n.history.clone()),
                min_activation_threshold: n.min_activation_threshold,
                bandwidth: 0.1,
                activity_ema: 0.0,
                last_spike_step: 0,
            })
            .collect();
        self.synapses = snapshot
            .synapses
            .iter()
            .map(|s| DelaySynapse {
                from: s.from,
                to: s.to,
                delay_ms: s.delay_ms,
                inhibitory: s.inhibitory,
                weight: if s.inhibitory { -1.0 } else { 1.0 },
                ring_buffer: VecDeque::from(s.ring_buffer.clone()),
            })
            .collect();
        self.time_step = snapshot.time_step;
        self.sample_rate_hz = snapshot.sample_rate_hz;
        self.adaptive_sensitivity = snapshot.adaptive_sensitivity;
        self.global_sensitivity = snapshot.global_sensitivity;
        self.input_energy_ema = snapshot.input_energy_ema;
        self.agc_rate = snapshot.agc_rate;
        self.rng_state = snapshot.rng_state;
        self.initial_seed = snapshot.initial_seed;
        self.privacy_mode = snapshot.privacy_mode;
        self.last_telemetry_step = snapshot.last_telemetry_step;
        self.input_bin_window.clear();
        self.input_bin_counts = [0; 16];
        Ok(())
    }

    /// Creates a new brain directly from a snapshot.
    pub fn from_snapshot(snapshot: &BrainSnapshot) -> Result<Self, CricketError> {
        let mut brain = CricketBrain::new(BrainConfig::default())?;
        brain.restore_from_snapshot(snapshot)?;
        Ok(brain)
    }

    fn snapshot_checksum(snapshot: &BrainSnapshot) -> u64 {
        fn mix(mut h: u64, bytes: &[u8]) -> u64 {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211);
            }
            h
        }
        let mut hash = 1469598103934665603u64;
        hash = mix(hash, snapshot.version_hash.as_bytes());
        hash = mix(hash, &snapshot.time_step.to_le_bytes());
        hash = mix(hash, &snapshot.sample_rate_hz.to_le_bytes());
        hash = mix(hash, &(snapshot.adaptive_sensitivity as u8).to_le_bytes());
        hash = mix(hash, &snapshot.global_sensitivity.to_le_bytes());
        hash = mix(hash, &snapshot.input_energy_ema.to_le_bytes());
        hash = mix(hash, &snapshot.agc_rate.to_le_bytes());
        hash = mix(hash, &snapshot.rng_state.to_le_bytes());
        hash = mix(hash, &snapshot.initial_seed.to_le_bytes());
        hash = mix(hash, &(snapshot.privacy_mode as u8).to_le_bytes());
        hash = mix(hash, &snapshot.last_telemetry_step.to_le_bytes());
        for n in &snapshot.neurons {
            hash = mix(hash, &n.id.to_le_bytes());
            hash = mix(hash, &n.eigenfreq.to_le_bytes());
            hash = mix(hash, &n.phase.to_le_bytes());
            hash = mix(hash, &n.amplitude.to_le_bytes());
            hash = mix(hash, &n.threshold.to_le_bytes());
            hash = mix(hash, &n.delay_taps.to_le_bytes());
            hash = mix(hash, &n.min_activation_threshold.to_le_bytes());
            for v in &n.history {
                hash = mix(hash, &v.to_le_bytes());
            }
        }
        for s in &snapshot.synapses {
            hash = mix(hash, &s.from.to_le_bytes());
            hash = mix(hash, &s.to.to_le_bytes());
            hash = mix(hash, &s.delay_ms.to_le_bytes());
            hash = mix(hash, &(s.inhibitory as u8).to_le_bytes());
            for v in &s.ring_buffer {
                hash = mix(hash, &v.to_le_bytes());
            }
        }
        hash
    }

    /// Returns a breakdown and total RAM estimate for this brain.
    pub fn total_ram_estimate(&self) -> BrainMemorySummary {
        let neuron_stats = self
            .neurons
            .iter()
            .map(Neuron::calculate_memory_requirements)
            .fold(MemoryStats::default(), |acc, s| MemoryStats {
                static_bytes: acc.static_bytes + s.static_bytes,
                dynamic_bytes: acc.dynamic_bytes + s.dynamic_bytes,
            });

        let synapse_stats = self
            .synapses
            .iter()
            .map(DelaySynapse::calculate_memory_requirements)
            .fold(MemoryStats::default(), |acc, s| MemoryStats {
                static_bytes: acc.static_bytes + s.static_bytes,
                dynamic_bytes: acc.dynamic_bytes + s.dynamic_bytes,
            });

        BrainMemorySummary {
            neuron_stats,
            synapse_stats,
        }
    }
}

impl Default for CricketBrain {
    fn default() -> Self {
        Self::new(BrainConfig::default()).expect("default BrainConfig must be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_brain() {
        let brain = CricketBrain::new(BrainConfig::default()).unwrap();
        assert_eq!(brain.neurons.len(), 5);
        assert_eq!(brain.synapses.len(), 6);
    }

    #[test]
    fn test_scaled_brain() {
        let brain = CricketBrain::new_scaled(100, 300).unwrap();
        assert_eq!(brain.neurons.len(), 100);
        assert_eq!(brain.synapses.len(), 300);
    }

    #[test]
    fn test_step_silence() {
        let mut brain = CricketBrain::new(BrainConfig::default()).unwrap();
        let out = brain.step(0.0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_reset() {
        let mut brain = CricketBrain::new(BrainConfig::default()).unwrap();
        brain.step(4500.0);
        brain.reset();
        assert_eq!(brain.time_step, 0);
        assert_eq!(brain.neurons[0].amplitude, 0.0);
    }

    #[test]
    fn test_deterministic_seed_reproducibility() {
        let cfg = BrainConfig::default().with_seed(42);
        let mut a = CricketBrain::new(cfg.clone()).unwrap();
        let mut b = CricketBrain::new(cfg).unwrap();
        let inputs = [4500.0, 0.0, 4500.0, 4400.0, 0.0, 4600.0];

        for &input in &inputs {
            let oa = a.step(input);
            let ob = b.step(input);
            assert!((oa - ob).abs() < 1e-9);
        }
    }

    #[test]
    fn test_reset_rewinds_seeded_rng() {
        let mut brain = CricketBrain::new(BrainConfig::default().with_seed(7)).unwrap();
        let inputs = [4500.0, 0.0, 4500.0, 4600.0];
        let first: Vec<f32> = inputs.iter().map(|&x| brain.step(x)).collect();
        brain.reset();
        let second: Vec<f32> = inputs.iter().map(|&x| brain.step(x)).collect();
        assert_eq!(first, second);
    }
}
