// SPDX-License-Identifier: AGPL-3.0-only
use cricket_brain::brain::{BrainConfig, CricketBrain};
use cricket_brain::error_codes;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryEvent {
    pub kind: String,
    pub value: f32,
    pub step: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictionSnapshot {
    pub confidence: f32,
    pub snr: f32,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorCodeMap {
    pub ok: i32,
    pub null: i32,
    pub invalid_config: i32,
    pub token_not_found: i32,
    pub invalid_input: i32,
    pub internal: i32,
}

#[wasm_bindgen]
pub struct Brain {
    inner: CricketBrain,
    events: Vec<TelemetryEvent>,
    prediction: PredictionSnapshot,
}

#[wasm_bindgen]
impl Brain {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: Option<u64>) -> Result<Brain, JsValue> {
        let config = BrainConfig::default().with_seed(seed.unwrap_or(12));
        let inner = CricketBrain::new(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self {
            inner,
            events: Vec::new(),
            prediction: PredictionSnapshot {
                confidence: 0.0,
                snr: 0.0,
                active: false,
            },
        })
    }

    pub fn step(&mut self, input_freq: f32) -> f32 {
        let out = self.inner.step(input_freq);
        if out > 0.0 {
            self.events.push(TelemetryEvent {
                kind: "spike".to_string(),
                value: out,
                step: self.inner.time_step,
            });
        }

        // Estimate SNR from neuron amplitudes: ratio of output (signal) to
        // mean resting activity (noise).  This replaces the previous hardcoded
        // placeholder and gives browser consumers a meaningful quality metric.
        let mean_amp = self
            .inner
            .neurons
            .iter()
            .map(|n| n.resonance_level())
            .sum::<f32>()
            / self.inner.neurons.len().max(1) as f32;
        let snr = if mean_amp > 1e-6 {
            out / mean_amp
        } else if out > 0.0 {
            40.0 // strong signal, no noise floor
        } else {
            0.0
        };
        // Confidence derived from SNR: saturates toward 1.0 for high SNR.
        let confidence = if snr > 0.0 {
            (snr / (1.0 + snr)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.prediction = PredictionSnapshot {
            confidence,
            snr,
            active: out > 0.0,
        };

        out
    }

    pub fn reset(&mut self) {
        self.inner.reset();
        self.events.clear();
    }

    pub fn time_step(&self) -> usize {
        self.inner.time_step
    }

    #[wasm_bindgen(js_name = drainTelemetry)]
    pub fn drain_telemetry(&mut self) -> Result<JsValue, JsValue> {
        let drained = self.events.clone();
        self.events.clear();
        serde_wasm_bindgen::to_value(&drained).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = latestPrediction)]
    pub fn latest_prediction(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.prediction)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = errorCodes)]
    pub fn error_codes() -> Result<JsValue, JsValue> {
        let map = ErrorCodeMap {
            ok: error_codes::CRICKET_OK,
            null: error_codes::CRICKET_ERR_NULL,
            invalid_config: error_codes::CRICKET_ERR_INVALID_CONFIG,
            token_not_found: error_codes::CRICKET_ERR_TOKEN_NOT_FOUND,
            invalid_input: error_codes::CRICKET_ERR_INVALID_INPUT,
            internal: error_codes::CRICKET_ERR_INTERNAL,
        };
        serde_wasm_bindgen::to_value(&map).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
