// SPDX-License-Identifier: AGPL-3.0-only
use alloc::collections::VecDeque;
use core::mem;

use crate::memory::{MemoryStats, EMBEDDED_RAM_LIMIT_BYTES, SAMPLE_BYTES};

/// Default delay used for embedded sizing checks.
pub const DEFAULT_SYNAPSE_DELAY_MS: usize = 5;
const DEFAULT_RING_BUFFER_BYTES: usize = DEFAULT_SYNAPSE_DELAY_MS * SAMPLE_BYTES;
const _: [(); EMBEDDED_RAM_LIMIT_BYTES - DEFAULT_RING_BUFFER_BYTES] =
    [(); EMBEDDED_RAM_LIMIT_BYTES - DEFAULT_RING_BUFFER_BYTES];

/// A delay synapse connecting two neurons with a fixed propagation delay.
///
/// Models the axonal delay lines found in the cricket auditory system.
/// Each synapse maintains a ring buffer that delays the signal by exactly
/// `delay_ms` timesteps before delivery.
///
/// In discrete time, let \(x[t]\) be the input signal and \(d\) the configured
/// delay in timesteps. The delayed output is:
///
/// \[
/// y[t] = x[t-d]
/// \]
///
/// The ring buffer stores the most recent \(d\) samples so each transmission
/// performs:
///
/// \[
/// y[t] \leftarrow \text{pop\_front}(B), \quad \text{push\_back}(B, x[t])
/// \]
///
/// Synapses can be excitatory (positive signal) or inhibitory (inverted signal).
#[derive(Debug, Clone)]
pub struct DelaySynapse {
    /// Source neuron index.
    pub from: usize,
    /// Target neuron index.
    pub to: usize,
    /// Propagation delay in milliseconds (timesteps).
    pub delay_ms: usize,
    /// If true, the synapse inverts the signal (inhibitory).
    pub inhibitory: bool,
    /// Synaptic weight (default: 1.0 excitatory, -1.0 inhibitory).
    /// Positive = excitatory, negative = inhibitory. Used by `transmit()`.
    /// Can be adjusted for plasticity (STDP, homeostasis).
    pub weight: f32,
    /// Ring buffer implementing the delay line.
    /// Length = delay_ms. The oldest element is at index 0.
    pub ring_buffer: VecDeque<f32>,
}

impl DelaySynapse {
    /// Creates a new delay synapse.
    ///
    /// # Arguments
    /// * `from` - Index of the source neuron
    /// * `to` - Index of the target neuron
    /// * `delay` - Propagation delay in ms (number of timesteps)
    /// * `inhibitory` - Whether this synapse inverts the signal
    ///
    /// # Example
    /// ```
    /// use cricket_brain_core::synapse::DelaySynapse;
    /// let syn = DelaySynapse::new(0, 1, 3, true);
    /// assert_eq!(syn.delay_ms, 3);
    /// assert!(syn.inhibitory);
    /// ```
    pub fn new(from: usize, to: usize, delay: usize, inhibitory: bool) -> Self {
        debug_assert!(delay > 0, "delay must be non-zero");
        let mut ring_buffer = VecDeque::with_capacity(delay);
        for _ in 0..delay {
            ring_buffer.push_back(0.0);
        }
        let weight = if inhibitory { -1.0 } else { 1.0 };
        DelaySynapse {
            from,
            to,
            delay_ms: delay,
            inhibitory,
            weight,
            ring_buffer,
        }
    }

    /// Transmits a signal through the delay line.
    ///
    /// The signal is delayed by exactly `delay_ms` timesteps using a ring buffer.
    /// For inhibitory synapses, the output is negated.
    ///
    /// ```text
    /// output(t) = buffer[t - delay_ms]
    /// buffer[t] = signal  (pushed to back)
    /// ```
    ///
    /// # Arguments
    /// * `signal` - The input signal amplitude from the source neuron
    ///
    /// # Returns
    /// The delayed (and possibly inverted) signal.
    #[inline(always)]
    pub fn transmit(&mut self, signal: f32) -> f32 {
        debug_assert!(self.delay_ms > 0, "delay must be non-zero");
        debug_assert!(
            self.ring_buffer.len() == self.delay_ms,
            "ring buffer length must match delay",
        );
        // BUG #3 FIX: Read the oldest element BEFORE pop_front.
        // This ensures we read the signal from exactly delay_ms steps ago.
        let delayed_output = self.ring_buffer[0];

        // Shift the buffer: remove oldest, add newest
        self.ring_buffer.pop_front();
        self.ring_buffer.push_back(signal);

        delayed_output * self.weight
    }

    /// Adjusts the synaptic weight by `delta`, clamped to `[-2.0, 2.0]`.
    ///
    /// Used by plasticity rules (STDP, homeostasis) to strengthen or weaken
    /// connections over time.
    ///
    /// # Arguments
    /// * `delta` - Amount to add to the current weight (positive = potentiate, negative = depress)
    #[inline]
    pub fn adjust_weight(&mut self, delta: f32) {
        self.weight = (self.weight + delta).clamp(-2.0, 2.0);
    }

    /// Returns the current synaptic weight.
    #[inline]
    pub fn current_weight(&self) -> f32 {
        self.weight
    }

    /// Returns the configured delay in timesteps.
    #[inline]
    pub fn delay(&self) -> usize {
        self.delay_ms
    }

    /// Returns the current number of samples stored in the delay line.
    #[inline]
    pub fn buffer_occupancy(&self) -> usize {
        self.ring_buffer.len()
    }

    /// Returns a read-only view of the internal delay ring buffer.
    #[inline]
    pub fn ring_buffer(&self) -> &VecDeque<f32> {
        &self.ring_buffer
    }

    /// Estimates memory requirements for this delay synapse.
    ///
    /// - `static_bytes`: `size_of::<DelaySynapse>()`
    /// - `dynamic_bytes`: ring-buffer length × `size_of::<f32>()`
    #[inline]
    pub fn calculate_memory_requirements(&self) -> MemoryStats {
        MemoryStats {
            static_bytes: mem::size_of::<Self>(),
            dynamic_bytes: self.ring_buffer.len() * mem::size_of::<f32>(),
        }
    }
}
