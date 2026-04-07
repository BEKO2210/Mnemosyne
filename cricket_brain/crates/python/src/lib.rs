// SPDX-License-Identifier: AGPL-3.0-only
use cricket_brain::brain::{BrainConfig, CricketBrain};
use cricket_brain::error_codes;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "BrainConfig")]
#[derive(Debug, Clone)]
pub struct PyBrainConfig {
    #[pyo3(get, set)]
    pub n_neurons: usize,
    #[pyo3(get, set)]
    pub min_freq: f32,
    #[pyo3(get, set)]
    pub max_freq: f32,
    #[pyo3(get, set)]
    pub sample_rate_hz: u32,
    #[pyo3(get, set)]
    pub min_activation_threshold: f32,
    #[pyo3(get, set)]
    pub adaptive_sensitivity: bool,
    #[pyo3(get, set)]
    pub agc_rate: f32,
    #[pyo3(get, set)]
    pub seed: u64,
    #[pyo3(get, set)]
    pub privacy_mode: bool,
}

impl Default for PyBrainConfig {
    fn default() -> Self {
        let cfg = BrainConfig::default();
        Self {
            n_neurons: cfg.n_neurons,
            min_freq: cfg.min_freq,
            max_freq: cfg.max_freq,
            sample_rate_hz: cfg.sample_rate_hz,
            min_activation_threshold: cfg.min_activation_threshold,
            adaptive_sensitivity: cfg.adaptive_sensitivity,
            agc_rate: cfg.agc_rate,
            seed: cfg.seed,
            privacy_mode: cfg.privacy_mode,
        }
    }
}

impl From<PyBrainConfig> for BrainConfig {
    fn from(value: PyBrainConfig) -> Self {
        BrainConfig {
            n_neurons: value.n_neurons,
            min_freq: value.min_freq,
            max_freq: value.max_freq,
            k_connections: None,
            sample_rate_hz: value.sample_rate_hz,
            min_activation_threshold: value.min_activation_threshold,
            adaptive_sensitivity: value.adaptive_sensitivity,
            agc_rate: value.agc_rate,
            seed: value.seed,
            privacy_mode: value.privacy_mode,
        }
    }
}

#[pymethods]
impl PyBrainConfig {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

#[pyclass(name = "Brain")]
pub struct PyBrain {
    inner: CricketBrain,
}

#[pymethods]
impl PyBrain {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyBrainConfig>) -> PyResult<Self> {
        let cfg = config.unwrap_or_default().into();
        let inner = CricketBrain::new(cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn step(&mut self, input_freq: f32) -> f32 {
        self.inner.step(input_freq)
    }

    fn step_batch(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.inner.step_batch(&inputs)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn time_step(&self) -> usize {
        self.inner.time_step
    }
}

#[pymodule(name = "cricket_brain")]
fn py_cricket_brain(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBrainConfig>()?;
    m.add_class::<PyBrain>()?;
    m.add("ERR_OK", error_codes::CRICKET_OK)?;
    m.add("ERR_NULL", error_codes::CRICKET_ERR_NULL)?;
    m.add(
        "ERR_INVALID_CONFIG",
        error_codes::CRICKET_ERR_INVALID_CONFIG,
    )?;
    m.add(
        "ERR_TOKEN_NOT_FOUND",
        error_codes::CRICKET_ERR_TOKEN_NOT_FOUND,
    )?;
    m.add("ERR_INVALID_INPUT", error_codes::CRICKET_ERR_INVALID_INPUT)?;
    m.add("ERR_INTERNAL", error_codes::CRICKET_ERR_INTERNAL)?;
    Ok(())
}
