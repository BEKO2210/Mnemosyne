// SPDX-License-Identifier: AGPL-3.0-only
//! Parallel resonator bank for multi-frequency token detection (v0.2).
//!
//! A [`ResonatorBank`] contains N parallel 5-neuron circuits (mini cricket-brains),
//! each tuned to a different token frequency. When a signal comes in, only the
//! matching bank fires — enabling real-time multi-token discrimination.

use crate::neuron::Neuron;
use crate::synapse::DelaySynapse;
use crate::token::TokenVocabulary;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::mem;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A single resonator channel — a 5-neuron cricket circuit tuned to one frequency.
#[derive(Debug, Clone)]
pub struct ResonatorChannel {
    /// Token ID this channel is tuned to.
    pub token_id: usize,
    /// The 5 neurons: [AN1, LN2, LN3, LN5, ON1].
    pub neurons: [Neuron; 5],
    /// The 6 synapses connecting the neurons.
    pub synapses: [DelaySynapse; 6],
}

impl ResonatorChannel {
    /// Creates a new resonator channel tuned to the given frequency.
    ///
    /// `bandwidth` controls the Gaussian selectivity (default 0.1 = 10%).
    /// For dense vocabularies, use a narrower value to avoid cross-activation.
    fn new(token_id: usize, freq: f32, base_neuron_id: usize, bandwidth: f32) -> Self {
        let mut neurons = [
            Neuron::new(base_neuron_id, freq, 4),     // AN1
            Neuron::new(base_neuron_id + 1, freq, 3), // LN2
            Neuron::new(base_neuron_id + 2, freq, 2), // LN3
            Neuron::new(base_neuron_id + 3, freq, 5), // LN5
            Neuron::new(base_neuron_id + 4, freq, 4), // ON1
        ];
        for n in &mut neurons {
            n.bandwidth = bandwidth;
        }

        let synapses = [
            DelaySynapse::new(0, 1, 3, true),  // AN1 → LN2 (inh)
            DelaySynapse::new(0, 2, 2, false), // AN1 → LN3 (exc)
            DelaySynapse::new(0, 3, 5, true),  // AN1 → LN5 (inh)
            DelaySynapse::new(1, 4, 1, true),  // LN2 → ON1 (inh)
            DelaySynapse::new(2, 4, 1, false), // LN3 → ON1 (exc)
            DelaySynapse::new(3, 4, 1, true),  // LN5 → ON1 (inh)
        ];

        ResonatorChannel {
            token_id,
            neurons,
            synapses,
        }
    }

    /// Processes one timestep with the given input frequency.
    /// Returns the output neuron (ON1) amplitude if coincidence fires, else 0.
    fn step(&mut self, input_freq: f32) -> f32 {
        let input_phase = 0.5; // fixed phase for token processing
        let is_silence = input_freq <= 0.0;

        // AN1 resonates
        if is_silence {
            self.neurons[0].amplitude *= 0.5;
            self.neurons[0].phase *= 0.5;
            let n0 = &mut self.neurons[0];
            if n0.history.len() == n0.delay_taps + 1 {
                n0.history.pop_front();
            }
            let amp = n0.amplitude;
            n0.history.push_back(amp);
        } else {
            self.neurons[0].resonate(input_freq, input_phase);
        }

        // Propagate through synapses
        let mut incoming = [0.0_f32; 5];
        for syn in &mut self.synapses {
            let src_amp = self.neurons[syn.from].amplitude;
            let out = syn.transmit(src_amp);
            incoming[syn.to] += out;
        }

        // Update downstream neurons
        #[allow(clippy::needless_range_loop)]
        for i in 1..5 {
            if is_silence {
                self.neurons[i].amplitude *= 0.5;
                self.neurons[i].phase *= 0.5;
                let ni = &mut self.neurons[i];
                if ni.history.len() == ni.delay_taps + 1 {
                    ni.history.pop_front();
                }
                let amp = ni.amplitude;
                ni.history.push_back(amp);
            } else {
                let signal = incoming[i];
                if signal.abs() > 0.01 {
                    self.neurons[i].resonate(input_freq, input_phase);
                    self.neurons[i].amplitude =
                        (self.neurons[i].amplitude + signal * 0.2).clamp(0.0, 1.0);
                } else {
                    self.neurons[i].decay();
                }
            }
        }

        // Check ON1 coincidence
        if self.neurons[4].check_coincidence() {
            self.neurons[4].amplitude
        } else {
            0.0
        }
    }

    /// Resets all neuron and synapse state.
    fn reset(&mut self) {
        for n in &mut self.neurons {
            n.amplitude = 0.0;
            n.phase = 0.0;
            for v in n.history.iter_mut() {
                *v = 0.0;
            }
        }
        for s in &mut self.synapses {
            for v in s.ring_buffer.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

/// A parallel bank of resonator channels for multi-token frequency detection.
///
/// Each channel is a complete 5-neuron cricket circuit tuned to one token's
/// frequency. On each timestep, all channels process the input in parallel,
/// and only the channel(s) matching the input frequency fire.
///
/// # Example
/// ```
/// use cricket_brain::token::TokenVocabulary;
/// use cricket_brain::resonator_bank::ResonatorBank;
///
/// let vocab = TokenVocabulary::from_labels(&["hello", "world"]);
/// let mut bank = ResonatorBank::new(&vocab);
///
/// // Feed the frequency of "hello" for 20 steps
/// let hello_freq = vocab.get("hello").unwrap().freq;
/// for _ in 0..20 {
///     let activations = bank.step(hello_freq);
///     // activations[0] ("hello") should be high
///     // activations[1] ("world") should be 0
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ResonatorBank {
    /// One resonator channel per token.
    pub channels: Vec<ResonatorChannel>,
    /// Current timestep.
    pub time_step: usize,
}

impl ResonatorBank {
    /// Creates a resonator bank from a vocabulary.
    ///
    /// Each token in the vocabulary gets its own 5-neuron resonator channel.
    /// Total neurons = `vocab.len() * 5`, total synapses = `vocab.len() * 6`.
    pub fn new(vocab: &TokenVocabulary) -> Self {
        // Compute adaptive bandwidth: half the minimum relative spacing between
        // adjacent tokens, clamped to [0.01, 0.10].  This ensures that the
        // Gaussian tuning curve of one channel does not significantly overlap
        // with its neighbors, even for dense vocabularies.
        let bandwidth = if vocab.tokens.len() > 1 {
            let min_relative_spacing = vocab
                .tokens
                .windows(2)
                .map(|w| (w[1].freq - w[0].freq).abs() / w[0].freq)
                .fold(f32::MAX, f32::min);
            (min_relative_spacing * 0.45).clamp(0.01, 0.10)
        } else {
            0.10
        };

        let channels = vocab
            .tokens
            .iter()
            .map(|token| ResonatorChannel::new(token.id, token.freq, token.id * 5, bandwidth))
            .collect();

        ResonatorBank {
            channels,
            time_step: 0,
        }
    }

    /// Processes one timestep. Returns activation amplitude for each token.
    ///
    /// The returned vector has one entry per token in the vocabulary.
    /// Values > 0.0 indicate the corresponding token's frequency was detected.
    pub fn step(&mut self, input_freq: f32) -> Vec<f32> {
        self.time_step += 1;
        #[cfg(feature = "parallel")]
        {
            self.channels
                .par_iter_mut()
                .map(|ch| ch.step(input_freq))
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.channels
                .iter_mut()
                .map(|ch| ch.step(input_freq))
                .collect()
        }
    }

    /// Processes one timestep and returns the ID of the most active token,
    /// or `None` if no channel fires.
    pub fn step_detect(&mut self, input_freq: f32) -> Option<usize> {
        let activations = self.step(input_freq);
        let (max_id, &max_val) = activations
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))?;
        if max_val > 0.0 {
            Some(max_id)
        } else {
            None
        }
    }

    /// Resets all channels.
    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
        self.time_step = 0;
    }

    /// Total number of neurons across all channels.
    pub fn total_neurons(&self) -> usize {
        self.channels.len() * 5
    }

    /// Total number of synapses across all channels.
    pub fn total_synapses(&self) -> usize {
        self.channels.len() * 6
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.channels
            .iter()
            .map(|ch| {
                let neuron_mem: usize = ch
                    .neurons
                    .iter()
                    .map(|n| mem::size_of::<Neuron>() + n.history.len() * 4)
                    .sum();
                let synapse_mem: usize = ch
                    .synapses
                    .iter()
                    .map(|s| mem::size_of::<DelaySynapse>() + s.ring_buffer.len() * 4)
                    .sum();
                neuron_mem + synapse_mem
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bank_creation() {
        let vocab = TokenVocabulary::from_labels(&["A", "B", "C"]);
        let bank = ResonatorBank::new(&vocab);
        assert_eq!(bank.channels.len(), 3);
        assert_eq!(bank.total_neurons(), 15);
        assert_eq!(bank.total_synapses(), 18);
    }

    #[test]
    fn test_bank_discrimination() {
        let vocab = TokenVocabulary::new(&["low", "high"], 2000.0, 6000.0);
        let mut bank = ResonatorBank::new(&vocab);

        // Feed "low" frequency for 20 steps
        let low_freq = vocab.get("low").unwrap().freq;
        let mut low_fired = false;
        let mut high_fired = false;
        for _ in 0..20 {
            let act = bank.step(low_freq);
            if act[0] > 0.0 {
                low_fired = true;
            }
            if act[1] > 0.0 {
                high_fired = true;
            }
        }

        assert!(low_fired, "Low channel should fire for low frequency");
        assert!(
            !high_fired,
            "High channel should NOT fire for low frequency"
        );
    }

    #[test]
    fn test_bank_silence() {
        let vocab = TokenVocabulary::from_labels(&["A", "B"]);
        let mut bank = ResonatorBank::new(&vocab);

        for _ in 0..20 {
            let act = bank.step(0.0);
            assert_eq!(act[0], 0.0);
            assert_eq!(act[1], 0.0);
        }
    }
}
