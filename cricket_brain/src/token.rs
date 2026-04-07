// SPDX-License-Identifier: AGPL-3.0-only
//! Multi-frequency token recognition (v0.2).
//!
//! Maps discrete symbols (characters, words, concepts) to unique frequencies.
//! A [`ResonatorBank`] contains parallel 5-neuron circuits, each tuned to one
//! token's frequency — enabling simultaneous multi-token detection.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::HashMap;

/// A token maps a symbolic label to a unique carrier frequency.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// Human-readable label (e.g., "A", "hello", "verb").
    pub label: String,
    /// Unique carrier frequency in Hz.
    pub freq: f32,
    /// Token index in the vocabulary.
    pub id: usize,
}

/// A vocabulary mapping symbols to frequency-encoded tokens.
///
/// Frequencies are assigned uniformly across a band (default: 2000–8000 Hz)
/// with guaranteed minimum spacing to prevent cross-activation between tokens.
///
/// # Example
/// ```
/// use cricket_brain::token::TokenVocabulary;
/// let vocab = TokenVocabulary::from_alphabet();
/// let token = vocab.get("A").unwrap();
/// assert!(token.freq >= 2000.0 && token.freq <= 8000.0);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct TokenVocabulary {
    /// All tokens, indexed by their ID.
    pub tokens: Vec<Token>,
    /// Label → token ID mapping for fast lookup.
    #[cfg(feature = "std")]
    label_map: HashMap<String, usize>,
    /// Lower bound of the frequency band (Hz).
    pub freq_min: f32,
    /// Upper bound of the frequency band (Hz).
    pub freq_max: f32,
}

impl TokenVocabulary {
    /// Creates a vocabulary from a list of labels.
    ///
    /// Frequencies are assigned uniformly across `[freq_min, freq_max]`.
    /// Minimum spacing between adjacent tokens is `(freq_max - freq_min) / n`.
    ///
    /// # Arguments
    /// * `labels` - Symbol labels (e.g., `["A", "B", "C"]`)
    /// * `freq_min` - Lower frequency bound (Hz)
    /// * `freq_max` - Upper frequency bound (Hz)
    pub fn new(labels: &[&str], freq_min: f32, freq_max: f32) -> Self {
        let n = labels.len();
        let step = if n > 1 {
            (freq_max - freq_min) / (n - 1) as f32
        } else {
            0.0
        };

        let mut tokens = Vec::with_capacity(n);
        #[cfg(feature = "std")]
        let mut label_map = HashMap::with_capacity(n);

        for (i, &label) in labels.iter().enumerate() {
            let freq = freq_min + step * i as f32;
            tokens.push(Token {
                label: label.to_string(),
                freq,
                id: i,
            });
            #[cfg(feature = "std")]
            label_map.insert(label.to_string(), i);
        }

        TokenVocabulary {
            tokens,
            #[cfg(feature = "std")]
            label_map,
            freq_min,
            freq_max,
        }
    }

    /// Creates a standard A–Z + space vocabulary (27 tokens, 2000–8000 Hz).
    pub fn from_alphabet() -> Self {
        let mut labels: Vec<String> = ('A'..='Z').map(|c| c.to_string()).collect();
        labels.push(" ".to_string());
        let refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        Self::new(&refs, 2000.0, 8000.0)
    }

    /// Creates a vocabulary from any string slice of labels.
    ///
    /// # Arguments
    /// * `labels` - Arbitrary token labels
    pub fn from_labels(labels: &[&str]) -> Self {
        // Assign frequency band based on vocabulary size
        // Minimum 200 Hz per token for clean Gaussian separation
        let freq_min = 2000.0;
        let freq_max = (freq_min + labels.len() as f32 * 200.0).max(4000.0);
        Self::new(labels, freq_min, freq_max)
    }

    /// Looks up a token by label.
    pub fn get(&self, label: &str) -> Option<&Token> {
        #[cfg(feature = "std")]
        {
            self.label_map.get(label).map(|&id| &self.tokens[id])
        }

        #[cfg(not(feature = "std"))]
        {
            self.tokens.iter().find(|t| t.label == label)
        }
    }

    /// Looks up a token by ID.
    pub fn get_by_id(&self, id: usize) -> Option<&Token> {
        self.tokens.get(id)
    }

    /// Returns the number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Encodes a string as a sequence of `(frequency, duration_ms)` pairs.
    ///
    /// Each character becomes a token frequency held for `token_ms`,
    /// separated by silence gaps of `gap_ms`.
    ///
    /// # Arguments
    /// * `text` - Input text (characters must be in the vocabulary)
    /// * `token_ms` - Duration of each token signal in ms
    /// * `gap_ms` - Silence gap between tokens in ms
    ///
    /// # Returns
    /// A vector of `(frequency_hz, duration_ms)` pairs.
    pub fn encode_text(&self, text: &str, token_ms: usize, gap_ms: usize) -> Vec<(f32, usize)> {
        let upper = text.to_uppercase();
        let chars: Vec<char> = upper.chars().collect();
        let mut result = Vec::new();

        for (i, ch) in chars.iter().enumerate() {
            let label = ch.to_string();
            if let Some(token) = self.get(&label) {
                result.push((token.freq, token_ms));
            }
            // Gap between tokens (but not after the last one)
            if i + 1 < chars.len() {
                result.push((0.0, gap_ms));
            }
        }

        result
    }

    /// Returns the frequency spacing between adjacent tokens.
    pub fn freq_spacing(&self) -> f32 {
        if self.tokens.len() > 1 {
            (self.freq_max - self.freq_min) / (self.tokens.len() - 1) as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alphabet_vocab() {
        let vocab = TokenVocabulary::from_alphabet();
        assert_eq!(vocab.len(), 27); // A-Z + space
        let a = vocab.get("A").unwrap();
        assert_eq!(a.id, 0);
        assert!((a.freq - 2000.0).abs() < 0.1);
    }

    #[test]
    fn test_custom_vocab() {
        let vocab = TokenVocabulary::from_labels(&["cat", "dog", "bird"]);
        assert_eq!(vocab.len(), 3);
        assert!(vocab.get("cat").is_some());
        assert!(vocab.get("fish").is_none());
    }

    #[test]
    fn test_freq_spacing() {
        let vocab = TokenVocabulary::from_alphabet();
        let spacing = vocab.freq_spacing();
        // 27 tokens across 6000 Hz → ~230 Hz per token
        // Gaussian 10% bandwidth at 4500 Hz = 450 Hz
        // Adjacent tokens at 230 Hz spacing can overlap — this is intentional
        assert!(spacing > 100.0, "Spacing too small: {spacing}");
    }

    #[test]
    fn test_encode_text() {
        let vocab = TokenVocabulary::from_alphabet();
        let signal = vocab.encode_text("AB", 50, 20);
        assert_eq!(signal.len(), 3); // A, gap, B
        assert!(signal[0].0 > 0.0); // A frequency
        assert_eq!(signal[1].0, 0.0); // gap
        assert!(signal[2].0 > signal[0].0); // B frequency > A frequency
    }
}
