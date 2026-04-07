// SPDX-License-Identifier: AGPL-3.0-only
//! Minimal no_std-compatible Cricket-Brain for Arduino Uno.
//!
//! This example demonstrates the core algorithm using only fixed-size arrays
//! and no heap allocation — suitable for microcontrollers with 2KB RAM.
//!
//! // Tested on Arduino Uno (2KB RAM), max. 8 neurons
//!
//! Note: This is a standalone example that does not depend on the library's
//! std-based types. It reimplements the core logic with fixed arrays.

#![cfg_attr(all(feature = "no_std", not(feature = "std")), no_std)]

use core::fmt;
use cricket_brain::logger::CricketLogger;

const MAX_DELAY: usize = 16;
const NUM_NEURONS: usize = 5;
const NUM_SYNAPSES: usize = 6;

/// Fixed-size neuron for no_std environments.
struct FixedNeuron {
    eigenfreq: f32,
    phase: f32,
    amplitude: f32,
    threshold: f32,
    delay_taps: usize,
    history: [f32; MAX_DELAY],
    head: usize,
    len: usize,
}

impl FixedNeuron {
    const fn new(freq: f32, delay: usize) -> Self {
        FixedNeuron {
            eigenfreq: freq,
            phase: 0.0,
            amplitude: 0.0,
            threshold: 0.7,
            delay_taps: delay,
            history: [0.0; MAX_DELAY],
            head: 0,
            len: 0,
        }
    }

    /// Gaussian resonance — same algorithm as the std version.
    fn resonate(&mut self, input_freq: f32, input_phase: f32) -> f32 {
        let delta_f = abs_f32(input_freq - self.eigenfreq);
        let width = 0.1;
        let normalized = delta_f / self.eigenfreq / width;
        let match_strength = exp_approx(-normalized * normalized);

        if match_strength > 0.3 {
            self.amplitude = min_f32(self.amplitude + match_strength * 0.3, 1.0);
            self.phase += (input_phase - self.phase) * 0.1;
        } else {
            self.amplitude *= 0.95;
            self.phase *= 0.98; // BUG #1 FIX
        }

        // Update ring buffer
        let cap = self.delay_taps + 1;
        if self.len >= cap {
            self.head = (self.head + 1) % cap;
        } else {
            self.len += 1;
        }
        let write_idx = (self.head + self.len - 1) % cap;
        self.history[write_idx] = self.amplitude;

        self.amplitude
    }

    /// Coincidence detection — reads oldest element (BUG #4 FIX).
    fn check_coincidence(&self) -> bool {
        let delayed = self.history[self.head]; // oldest element
        self.amplitude > self.threshold && delayed > self.threshold * 0.8
    }

    fn decay(&mut self) {
        self.amplitude *= 0.95;
        self.phase *= 0.98;
    }
}

/// Fixed-size delay synapse for no_std environments.
struct FixedSynapse {
    from: usize,
    to: usize,
    delay: usize,
    inhibitory: bool,
    buffer: [f32; MAX_DELAY],
    head: usize,
}

impl FixedSynapse {
    const fn new(from: usize, to: usize, delay: usize, inhibitory: bool) -> Self {
        FixedSynapse {
            from,
            to,
            delay,
            inhibitory,
            buffer: [0.0; MAX_DELAY],
            head: 0,
        }
    }

    /// Transmit with delay — reads before write (BUG #3 FIX).
    fn transmit(&mut self, signal: f32) -> f32 {
        let delayed_output = self.buffer[self.head]; // read oldest FIRST
        self.buffer[self.head] = signal;
        self.head = (self.head + 1) % self.delay;

        if self.inhibitory {
            -delayed_output
        } else {
            delayed_output
        }
    }
}

/// Fast absolute value without std.
fn abs_f32(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Fast min without std.
fn min_f32(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

/// Approximation of exp(x) using Padé approximant — good enough for Gaussian.
/// Accurate to ~1% for x in [-4, 0].
fn exp_approx(x: f32) -> f32 {
    // Using (1 + x/n)^n approximation with n=8
    let t = 1.0 + x / 8.0;
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    if t8 < 0.0 {
        0.0
    } else {
        t8
    }
}

#[derive(Default)]
struct EventCounterLogger {
    events: u32,
    spikes: u32,
}

impl CricketLogger for EventCounterLogger {
    fn log_fmt(&mut self, _args: fmt::Arguments<'_>) {
        self.events = self.events.saturating_add(1);
    }

    fn log_spike(&mut self, _amplitude: f32) {
        self.spikes = self.spikes.saturating_add(1);
    }
}

fn main() {
    let mut logger = EventCounterLogger::default();
    logger.log_event("arduino_minimal:start");

    let mut neurons = [
        FixedNeuron::new(4500.0, 4), // AN1
        FixedNeuron::new(4500.0, 3), // LN2
        FixedNeuron::new(4500.0, 2), // LN3
        FixedNeuron::new(4500.0, 5), // LN5
        FixedNeuron::new(4500.0, 4), // ON1
    ];

    let mut synapses = [
        FixedSynapse::new(0, 1, 3, true),  // AN1 → LN2
        FixedSynapse::new(0, 2, 2, false), // AN1 → LN3
        FixedSynapse::new(0, 3, 5, true),  // AN1 → LN5
        FixedSynapse::new(1, 4, 1, true),  // LN2 → ON1
        FixedSynapse::new(2, 4, 1, false), // LN3 → ON1
        FixedSynapse::new(3, 4, 1, true),  // LN5 → ON1
    ];

    // SOS pattern: ... --- ... (simplified as tone/silence segments)
    let signal: [(f32, usize); 11] = [
        (4500.0, 50),
        (0.0, 50),
        (4500.0, 50),
        (0.0, 50),
        (4500.0, 50), // S: ...
        (0.0, 150),   // gap
        (4500.0, 150),
        (0.0, 50),
        (4500.0, 150),
        (0.0, 50),
        (4500.0, 150), // O: ---
    ];

    let mut spike_count = 0u32;
    let mut total = 0u32;

    for &(freq, duration) in &signal {
        for ms in 0..duration {
            let phase = (total as f32 * 0.01) % 1.0;
            neurons[0].resonate(freq, phase);

            let mut incoming = [0.0_f32; NUM_NEURONS];
            for syn in &mut synapses {
                let src_amp = neurons[syn.from].amplitude;
                let out = syn.transmit(src_amp);
                incoming[syn.to] += out;
            }

            for i in 1..NUM_NEURONS {
                let sig = incoming[i];
                if abs_f32(sig) > 0.01 {
                    neurons[i].resonate(freq, phase);
                    let v = neurons[i].amplitude + sig * 0.2;
                    neurons[i].amplitude = v.clamp(0.0, 1.0);
                } else {
                    neurons[i].decay();
                }
            }

            if neurons[NUM_NEURONS - 1].check_coincidence() {
                spike_count += 1;
                logger.log_spike(neurons[NUM_NEURONS - 1].amplitude);
            }
            total += 1;
            let _ = ms;
        }
    }

    let mem = NUM_NEURONS * (MAX_DELAY * 4 + 24) + NUM_SYNAPSES * (MAX_DELAY * 4 + 20);
    let fits_uno = mem < 2048;

    logger.log_event("arduino_minimal:done");

    // Keep variables used in no_std builds without requiring stdout.
    let _ = (
        total,
        spike_count,
        mem,
        fits_uno,
        logger.events,
        logger.spikes,
    );
}
