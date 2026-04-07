# CricketBrain: A Biomorphic Delay-Line Coincidence Detector for Real-Time Temporal Pattern Recognition

**Author:** Belkis Aslani

**Version:** 1.0.0 | **Date:** 2026-04-06

**Repository:** https://github.com/BEKO2210/cricket-brain

> **AI-Assisted Development:** This project was developed with systematic use of
> AI coding assistants (Claude Code, ChatGPT/Codex, Kimi, Gemini). See
> `AI_DEVELOPMENT_STATEMENT.md` for full disclosure.

---

## Abstract

We present CricketBrain, a neuromorphic signal processor inspired by the
auditory pathway of the field cricket (*Gryllus bimaculatus*). The system
employs Gaussian-tuned resonators, ring-buffer delay lines, and coincidence
detection gates to recognize temporal patterns in real-time with sub-microsecond
latency and sub-kilobyte memory footprints. On a canonical 5-neuron circuit
modeled after the cricket's AN1-LN2/LN3/LN5-ON1 pathway, we achieve 0.175 us
per processing step with 944 bytes of RAM in a `no_std` embedded configuration.
We evaluate the system across SNR conditions from -10 dB to +30 dB using a
parametric sweep protocol with 120 trials per condition, and compare against
three classical baselines: matched filtering, Goertzel-based spectral detection,
and IIR bandpass filtering. An ablation study demonstrates that each circuit
component contributes measurably to detection performance, with the coincidence
gate providing the largest single improvement in false-positive rejection.
The implementation is released as a Rust workspace with C, Python, and
WebAssembly bindings.

---

## 1. Introduction

Temporal pattern recognition in streaming signals — detecting specific rhythmic
structures embedded in noise — is a fundamental problem in biomedical monitoring
[1], industrial condition monitoring [2], and IoT security [3]. Conventional
approaches rely on spectral analysis (FFT), matched filtering, or deep learning,
each with characteristic trade-offs in latency, memory, and adaptability.

Biological auditory systems, particularly those of acoustically communicating
insects, have evolved remarkably efficient solutions to this problem. The field
cricket *Gryllus bimaculatus* can reliably identify conspecific calling songs
(~4.5 kHz carrier, species-specific pulse patterns) in noisy environments using
a neural circuit of fewer than 10 identified neurons [4, 5].

We draw direct inspiration from this circuit to build a computational model
that preserves the biological principles — resonant frequency selectivity,
axonal delay lines, and coincidence detection — while targeting modern embedded
and edge-computing platforms.

### Contributions

1. A complete, open-source implementation of a biomorphic delay-line coincidence
   detector in safe Rust, with `no_std` support for embedded deployment.
2. Quantitative comparison against three classical baselines showing favorable
   false-positive rejection at equivalent true-positive rates.
3. A systematic ablation study demonstrating the necessity of each circuit
   component.
4. Cross-platform bindings (C FFI, Python/PyO3, WebAssembly) enabling
   integration into diverse research and production environments.

---

## 2. Related Work

### 2.1 Cricket Auditory Neuroscience

The neural basis of phonotaxis in crickets has been studied extensively since
Huber's early work on central pattern generators [6]. Schildberger [7]
identified temporal filtering properties in the cricket brain that enable
discrimination of conspecific songs based on pulse rate. Schoeneich, Kostarakos,
and Hedwig [8] mapped the complete ascending auditory pathway, identifying the
AN1 receptor neuron, local brain neurons (LN2-LN5), and the descending ON1
output neuron as the core circuit for song recognition. Hedwig and Poulet [9]
demonstrated that this circuit implements a delay-line coincidence detection
mechanism, where inhibitory and excitatory pathways with different propagation
delays create a temporal filter tuned to the species-specific pulse interval.

### 2.2 Neuromorphic Engineering

Mead's foundational work on analog VLSI [10] established the field of
neuromorphic engineering. Subsequent developments in silicon neuron circuits
[11] and neuromorphic sensory systems [12] demonstrated that biological
principles can yield efficient hardware implementations. The SpiNNaker [13]
and Loihi [14] platforms have shown that spiking neural networks can be
executed with orders-of-magnitude improvements in energy efficiency compared
to conventional processors, particularly for temporal pattern tasks.

### 2.3 Classical Signal Detection

Matched filtering is the optimal linear detector under additive white Gaussian
noise [15] but requires exact template knowledge and is sensitive to timing
jitter. The Goertzel algorithm [16] provides efficient single-frequency
spectral analysis but lacks temporal structure sensitivity. IIR bandpass
filters offer continuous frequency selection but produce sustained output
during any in-band energy, yielding high false-positive rates for pattern
discrimination tasks.

---

## 3. Biological Background

The *Gryllus bimaculatus* male calling song consists of chirps at approximately
4.5 kHz carrier frequency, with species-specific pulse duration (~20 ms) and
inter-pulse interval (~40 ms) [4]. Female phonotaxis — directional movement
toward the sound source — requires recognition of both the carrier frequency
and the temporal pattern.

The neural circuit responsible for this recognition consists of:

- **AN1** (Ascending Neuron 1): Auditory receptor tuned to the carrier
  frequency via mechanical resonance of the tympanic membrane. Projects to
  the brain with high temporal fidelity [8].
- **LN2** (Local Neuron 2): Inhibitory interneuron receiving AN1 input with
  ~3 ms axonal delay. Provides delayed inhibition to the output stage.
- **LN3** (Local Neuron 3): Excitatory interneuron receiving AN1 input with
  ~2 ms delay. Provides the "reference" signal for coincidence detection.
- **LN5** (Local Neuron 5): Second inhibitory interneuron with ~5 ms delay.
  Implements a wider temporal rejection window.
- **ON1** (Output Neuron 1): Coincidence detector that fires only when
  excitatory input from LN3 arrives simultaneously with a gap in inhibition
  from LN2 and LN5. This creates a temporal bandpass for the pulse interval.

```
         AN1 (Receptor, 4500 Hz)
        / | \
       /  |  \
      v   v   v
    LN2  LN3  LN5
   (inh) (exc) (inh)
   3ms   2ms   5ms
      \   |   /
       \  |  /
        v v v
     ON1 (Output Gate)
```

The key insight is that the differential delays (2 ms, 3 ms, 5 ms) create
a temporal filter: ON1 fires maximally when pulses arrive at the
species-specific interval, because only then does the excitatory LN3 signal
coincide with gaps in the inhibitory LN2 and LN5 signals.

---

## 4. Mathematical Model

### 4.1 Gaussian Frequency Selectivity

Each neuron *i* has an eigenfrequency *f_0* and responds to input frequency
*f_in* according to a Gaussian tuning curve:

```
R(f_in, f_0) = exp( -(|f_in - f_0| / (f_0 * w))^2 )
```

where *w* is the bandwidth parameter (default 0.1 = 10% relative bandwidth;
configurable per-neuron via `Neuron.bandwidth`, and automatically adapted by
`ResonatorBank` based on token spacing for dense vocabularies). The resonance
threshold is set at
*R* > 0.3, yielding an effective bandwidth of approximately +-11% of the
eigenfrequency. This matches the frequency selectivity observed in cricket AN1
neurons [8].

### 4.2 Amplitude Dynamics

When resonating (*R* > 0.3):
```
A(t+1) = min( A(t) + R * 0.3, 1.0 )
```

When not resonating:
```
A(t+1) = A(t) * 0.95
```

The decay constant (tau = -1/ln(0.95) ~ 19.5 steps) ensures that brief noise
bursts do not sustain activation.

### 4.3 Phase Locking

During resonance, the neuron's internal phase locks to the input:
```
phi(t+1) = phi(t) + (phi_in - phi(t)) * 0.1
```

During silence, the phase drifts toward zero:
```
phi(t+1) = phi(t) * 0.98
```

### 4.4 Synaptic Delay

Each synapse implements a pure delay line via a FIFO ring buffer:
```
y(t) = x(t - d)        (excitatory)
y(t) = -x(t - d)       (inhibitory)
```

where *d* is the propagation delay in timesteps.

### 4.5 Coincidence Detection Gate

The output neuron ON1 fires only when both current and delayed evidence
exceed the threshold:
```
fire(t) = [ A(t) > theta ] AND [ A(t - tau) > theta * 0.8 ]
```

where *theta* = 0.7 (default firing threshold) and *tau* is the delay tap
length. The 0.8 factor on the delayed condition provides tolerance for
amplitude decay, following the biological observation that temporal integration
windows are slightly asymmetric [9].

### 4.6 Confidence Scoring

For sequence prediction, confidence is computed as:
```
C = clip( SNR / (1 + SNR) * (1 - J/T), 0, 1 )
```

where SNR is the instantaneous signal-to-noise ratio, *J* is temporal jitter
in milliseconds, and *T* is the configured tolerance window.

### 4.7 Computational Complexity

Per processing step:
- **Time:** O(N + S) where N = neurons, S = synapses
- **Space:** O(N * H + S * D) where H = history length, D = delay length

For the canonical 5-neuron circuit: N=5, S=6, H<=5, D<=5, yielding constant
per-step cost.

---

## 5. Architecture and Implementation

### 5.1 Workspace Structure

The implementation is organized as a Rust workspace:

| Crate | Purpose | `no_std` | LOC |
|-------|---------|----------|-----|
| `cricket-brain-core` | Neuron, synapse, memory, telemetry primitives | Yes | ~600 |
| `cricket-brain` | Brain network, sequence predictor, resonator bank | Optional | ~2,700 |
| `cricket-brain-ffi` | C-compatible API | No | ~165 |
| `cricket-brain-python` | PyO3 bindings | No | ~122 |
| `cricket-brain-wasm` | wasm-bindgen bindings | No | ~120 |

### 5.2 Safety Guarantees

- `#![deny(unsafe_code)]` in the core crate
- `unsafe` only in the FFI boundary layer, with documented safety invariants
- Zero external dependencies in core (only `libm` for `no_std` math)

### 5.3 Feature Flags

| Flag | Effect |
|------|--------|
| `std` (default) | Standard library support |
| `no_std` | Embedded mode (`alloc` only) |
| `serde` | Snapshot serialization with CRC64 checksums |
| `parallel` | Rayon-based resonator bank parallelism |
| `telemetry` | Structured event hooks |
| `cli` | JSON Lines telemetry sink |

---

## 6. Experimental Evaluation

### 6.1 Experimental Protocol

All experiments use a standardized trial structure:

- **Warm-up:** 24 timesteps of background noise
- **Observation window:** 120 timesteps; signal present in steps 32-92 (60 ms burst)
- **Signal:** 4500 Hz carrier with SNR-dependent frequency jitter
- **Noise:** Random frequency bursts with SNR-dependent probability
- **Trials:** 120 per class (signal-present and signal-absent) per condition
- **Determinism:** LCG-based PRNG with fixed seeds for reproducibility

### 6.2 Performance Metrics

| Configuration | Latency | Throughput | Memory |
|---|---|---|---|
| 5-neuron canonical | 0.175 us/step | 5.7 M steps/sec | 348 bytes |
| `no_std` Arduino minimal | N/A | N/A | 944 bytes (static) |
| 40,960-neuron scale | N/A | 3.43e7 neuron-ops/sec | 13.91 MB |
| Sequence predictor (1,280 neurons) | N/A | 3.32e7 neuron-ops/sec | 0.30 MB |

Benchmarks measured with Criterion on x86-64 (specific CPU and rustc version
recorded in benchmark artifacts).

### 6.3 Baseline Comparisons

Three classical detectors are evaluated under identical conditions:

1. **Matched Filter:** Normalized cross-correlation with a 60-sample 4500 Hz
   template. Optimal under AWGN assumptions [15].
2. **Goertzel Detector:** Single-frequency DFT at 4500 Hz using the Goertzel
   algorithm [16], with sliding-window magnitude thresholding.
3. **IIR Bandpass Filter:** Second-order IIR bandpass centered at 4500 Hz
   (Q ~ 5), with envelope detection via exponential smoothing.

Results are generated by `examples/baselines.rs` and archived at
`target/research/baseline_comparison.csv`.

### 6.4 Ablation Study

To demonstrate that each circuit component contributes to performance, we
systematically disable individual elements:

| Configuration | Modification |
|---|---|
| Full circuit (control) | No modification |
| Without LN2 | Disable AN1->LN2 and LN2->ON1 synapses |
| Without LN3 | Disable AN1->LN3 and LN3->ON1 synapses |
| Without LN5 | Disable AN1->LN5 and LN5->ON1 synapses |
| Without coincidence gate | Output based on ON1 amplitude only |
| Minimal delays | All synapse delays set to 1 ms |

Results are generated by `examples/ablation_study.rs` and archived at
`target/research/ablation_study.csv`.

### 6.5 Reproducibility

All experiments can be reproduced from the repository:

```bash
# Baselines comparison
cargo run --release --example baselines

# Ablation study
cargo run --release --example ablation_study

# Full SNR sweep (8 sensitivity levels x 9 SNR levels x 120 trials x 2 classes)
cargo run --release --example research_gen -- --output target/research --seed 1337
```

Deterministic seeding (`BrainConfig::with_seed(...)` and LCG PRNG) ensures
bitwise-identical results across platforms (verified in CI on Linux, macOS,
and Windows).

---

## 7. Discussion

### 7.1 Strengths

The biomorphic approach offers several advantages over classical methods:

- **Temporal selectivity:** The coincidence gate inherently rejects sustained
  in-band noise that would trigger bandpass-based detectors.
- **Minimal memory:** The ring-buffer delay implementation requires only
  O(D) storage per synapse, enabling deployment on microcontrollers.
- **Deterministic latency:** Processing time is constant per step with no
  data-dependent branching in the hot path.

### 7.2 Limitations

- **Fixed topology:** The circuit structure (5 neurons, 6 synapses) is fixed.
  However, synaptic weights and thresholds can now adapt online via STDP
  and homeostatic plasticity (opt-in, see `plasticity` module).
- **Single-carrier assumption:** The Gaussian tuning curve is centered on a
  single eigenfrequency. Multi-frequency patterns require the ResonatorBank
  extension.
- **No spike timing:** The model uses rate-coded amplitudes rather than
  precise spike timing, limiting temporal resolution to the timestep granularity.

### 7.3 Future Work

- ~~Spike-timing dependent plasticity (STDP)~~ — implemented in v3.0
  (`plasticity` module: `StdpConfig`, `HomeostasisConfig`, 37 tests)
- Hardware deployment on RISC-V and ARM Cortex-M targets
- Medical device certification (IEC 62304) for the ECG sentinel application
- Formal verification of the delay-line coincidence logic

---

## 8. Conclusion

CricketBrain demonstrates that biologically inspired delay-line coincidence
detection is a viable and efficient approach to temporal pattern recognition
in embedded systems. The system achieves sub-microsecond processing latency
and sub-kilobyte memory footprints while providing robust detection across a
wide range of SNR conditions. The open-source Rust implementation, with its
`no_std` core and multi-language bindings, is intended to serve as a foundation
for both research exploration and production deployment in medical monitoring,
industrial IoT, and edge-computing applications.

---

## References

[1] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet:
    Components of a new research resource for complex physiologic signals,"
    *Circulation*, vol. 101, no. 23, pp. e215-e220, 2000.

[2] R. B. Randall, *Vibration-Based Condition Monitoring: Industrial,
    Automotive and Aerospace Applications*, 2nd ed. Wiley, 2021.

[3] Y. Mirsky et al., "Kitsune: An ensemble of autoencoders for online
    network intrusion detection," in *Proc. NDSS*, 2018.

[4] R. R. Hoy, "The evolution of hearing in insects as an adaptation to
    predation from bats," in *The Evolutionary Biology of Hearing*,
    D. B. Webster, R. R. Fay, and A. N. Popper, Eds. Springer, 1992,
    pp. 115-129.

[5] T. Weber and J. Thorson, "Phonotactic behavior of walking crickets,"
    *J. Comp. Physiol. A*, vol. 163, pp. 495-502, 1988.

[6] F. Huber, "Neural correlates of orthopteran and cicada phonotaxis,"
    in *Neuroethology and Behavioral Physiology*, F. Huber and H. Markl,
    Eds. Springer, 1983, pp. 108-135.

[7] K. Schildberger, "Temporal selectivity of identified auditory neurons
    in the cricket brain," *J. Comp. Physiol. A*, vol. 155, pp. 171-185,
    1984.

[8] S. Schoeneich, K. Kostarakos, and B. Hedwig, "An auditory feature
    detection circuit for sound pattern recognition," *Science Advances*,
    vol. 1, no. 8, e1500325, 2015.

[9] B. Hedwig and J. F. A. Poulet, "Mechanisms underlying phonotactic
    steering in the cricket *Gryllus bimaculatus* revealed with a fast
    trackball system," *J. Exp. Biol.*, vol. 208, pp. 915-927, 2005.

[10] C. Mead, *Analog VLSI and Neural Systems*. Addison-Wesley, 1989.

[11] G. Indiveri et al., "Neuromorphic silicon neuron circuits," *Frontiers
     in Neuroscience*, vol. 5, art. 73, 2011.

[12] S.-C. Liu and T. Delbruck, "Neuromorphic sensory systems," *Current
     Opinion in Neurobiology*, vol. 20, no. 3, pp. 288-295, 2010.

[13] S. B. Furber et al., "The SpiNNaker project," *Proc. IEEE*, vol. 102,
     no. 5, pp. 652-665, 2014.

[14] M. Davies et al., "Loihi: A neuromorphic manycore processor with
     on-chip learning," *IEEE Micro*, vol. 38, no. 1, pp. 82-99, 2018.

[15] D. O. North, "An analysis of the factors which determine signal/noise
     discrimination in pulsed-carrier systems," *Proc. IEEE*, vol. 51,
     no. 7, pp. 1016-1027, 1963.

[16] G. Goertzel, "An algorithm for the evaluation of finite trigonometric
     series," *The American Mathematical Monthly*, vol. 65, no. 1,
     pp. 34-35, 1958.

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| f_0 | Neuron eigenfrequency (Hz) |
| f_in | Input frequency (Hz) |
| w | Bandwidth parameter (default 0.1 = 10%; adaptive in ResonatorBank) |
| R | Resonance response (0-1) |
| A(t) | Neuron amplitude at timestep t |
| phi(t) | Neuron phase at timestep t |
| theta | Firing threshold (default 0.7) |
| tau | Delay tap length (ms) |
| d | Synaptic propagation delay (ms) |
| C | Confidence score (0-1) |
| SNR | Signal-to-noise ratio |
| J | Temporal jitter (ms) |
| T | Temporal tolerance window (ms) |

## Appendix B: Reproducibility Checklist

- [ ] Clone repository at tagged version (v3.0.0)
- [ ] Verify Rust toolchain >= 1.75 (`rustup show`)
- [ ] Run `cargo test --workspace` (all tests pass)
- [ ] Run `cargo run --release --example research_gen -- --seed 1337`
- [ ] Run `cargo run --release --example baselines`
- [ ] Run `cargo run --release --example ablation_study`
- [ ] Compare CSV outputs against published values
- [ ] Record: CPU model, OS version, rustc version, optimization flags
