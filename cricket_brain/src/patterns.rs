// SPDX-License-Identifier: AGPL-3.0-only
//! Morse code and rhythm pattern utilities for the Cricket-Brain.
//!
//! Provides encoding of text into frequency/duration pairs suitable
//! for driving the CricketBrain, and decoding of spike outputs back
//! into text.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Carrier frequency for Morse signals (Hz).
pub const MORSE_FREQ: f32 = 4500.0;
/// Duration of a dot in milliseconds.
pub const DOT_MS: usize = 50;
/// Duration of a dash in milliseconds.
pub const DASH_MS: usize = 150;
/// Duration of an intra-character gap in milliseconds.
pub const ELEMENT_GAP_MS: usize = 50;
/// Duration of an inter-character gap in milliseconds.
pub const CHAR_GAP_MS: usize = 150;
/// Duration of a word gap in milliseconds.
pub const WORD_GAP_MS: usize = 350;

/// Represents a single Morse code element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MorseSymbol {
    /// Short signal (dit).
    Dot,
    /// Long signal (dah).
    Dash,
    /// Inter-element or inter-character gap (silence).
    Gap,
    /// Word separator gap (longer silence).
    WordGap,
}

/// Encodes a text string into a sequence of `(frequency, duration_ms)` pairs.
///
/// Letters are converted to Morse code, with appropriate gaps inserted.
/// During signal elements, frequency is `MORSE_FREQ`. During gaps, frequency is 0.0.
///
/// # Arguments
/// * `text` - ASCII text to encode (uppercase or lowercase)
///
/// # Returns
/// A vector of `(frequency_hz, duration_ms)` pairs.
///
/// # Example
/// ```
/// use cricket_brain::patterns::encode_morse;
/// let signal = encode_morse("SOS");
/// assert!(!signal.is_empty());
/// ```
pub fn encode_morse(text: &str) -> Vec<(f32, usize)> {
    let mut result = Vec::new();
    let upper = text.to_uppercase();
    let chars: Vec<char> = upper.chars().collect();

    for (ci, &ch) in chars.iter().enumerate() {
        if ch == ' ' {
            result.push((0.0, WORD_GAP_MS));
            continue;
        }

        let symbols = char_to_morse(ch);
        if symbols.is_empty() {
            continue;
        }

        for (si, &sym) in symbols.iter().enumerate() {
            match sym {
                MorseSymbol::Dot => {
                    result.push((MORSE_FREQ, DOT_MS));
                }
                MorseSymbol::Dash => {
                    result.push((MORSE_FREQ, DASH_MS));
                }
                MorseSymbol::Gap | MorseSymbol::WordGap => {}
            }
            // Intra-character gap (between elements of same character)
            if si + 1 < symbols.len() {
                result.push((0.0, ELEMENT_GAP_MS));
            }
        }

        // Inter-character gap
        if ci + 1 < chars.len() && chars[ci + 1] != ' ' {
            result.push((0.0, CHAR_GAP_MS));
        }
    }

    result
}

/// Decodes spike output from the CricketBrain back into text.
///
/// Groups spikes by timing to identify dots and dashes, then maps
/// Morse sequences back to characters.
///
/// # Arguments
/// * `spikes` - Slice of `(timestep, amplitude)` pairs from the brain output
/// * `threshold` - Minimum amplitude to count as an active spike
///
/// # Returns
/// The decoded text string.
pub fn decode_spikes(spikes: &[(usize, f32)], threshold: f32) -> String {
    if spikes.is_empty() {
        return String::new();
    }

    // Group contiguous above-threshold spikes into bursts
    let mut bursts: Vec<(usize, usize)> = Vec::new(); // (start, length)
    let mut in_burst = false;
    let mut burst_start = 0;
    let mut burst_len = 0;

    for &(t, amp) in spikes {
        if amp > threshold {
            if !in_burst {
                burst_start = t;
                burst_len = 1;
                in_burst = true;
            } else {
                burst_len += 1;
            }
        } else if in_burst {
            bursts.push((burst_start, burst_len));
            in_burst = false;
        }
    }
    if in_burst {
        bursts.push((burst_start, burst_len));
    }

    // Convert bursts to Morse symbols based on duration
    let dot_dash_threshold = (DOT_MS + DASH_MS) / 2;
    let mut morse_chars: Vec<Vec<MorseSymbol>> = Vec::new();
    let mut current_char: Vec<MorseSymbol> = Vec::new();

    for (i, &(start, len)) in bursts.iter().enumerate() {
        let sym = if len > dot_dash_threshold {
            MorseSymbol::Dash
        } else {
            MorseSymbol::Dot
        };
        current_char.push(sym);

        // Check gap to next burst for character/word boundaries
        if i + 1 < bursts.len() {
            let gap = bursts[i + 1].0 - (start + len);
            // Character boundary: gap >= inter-character gap duration
            if gap >= CHAR_GAP_MS {
                morse_chars.push(current_char.clone());
                current_char.clear();
                // Word boundary: gap >= word gap duration
                if gap >= WORD_GAP_MS {
                    morse_chars.push(vec![MorseSymbol::WordGap]);
                }
            }
        }
    }
    if !current_char.is_empty() {
        morse_chars.push(current_char);
    }

    // Convert Morse symbol sequences to characters
    let mut result = String::new();
    for symbols in &morse_chars {
        if symbols == &[MorseSymbol::WordGap] {
            result.push(' ');
        } else {
            result.push(morse_to_char(symbols));
        }
    }

    result
}

/// Maps a character to its Morse code representation.
fn char_to_morse(ch: char) -> Vec<MorseSymbol> {
    use MorseSymbol::{Dash, Dot};
    match ch {
        'A' => vec![Dot, Dash],
        'B' => vec![Dash, Dot, Dot, Dot],
        'C' => vec![Dash, Dot, Dash, Dot],
        'D' => vec![Dash, Dot, Dot],
        'E' => vec![Dot],
        'F' => vec![Dot, Dot, Dash, Dot],
        'G' => vec![Dash, Dash, Dot],
        'H' => vec![Dot, Dot, Dot, Dot],
        'I' => vec![Dot, Dot],
        'J' => vec![Dot, Dash, Dash, Dash],
        'K' => vec![Dash, Dot, Dash],
        'L' => vec![Dot, Dash, Dot, Dot],
        'M' => vec![Dash, Dash],
        'N' => vec![Dash, Dot],
        'O' => vec![Dash, Dash, Dash],
        'P' => vec![Dot, Dash, Dash, Dot],
        'Q' => vec![Dash, Dash, Dot, Dash],
        'R' => vec![Dot, Dash, Dot],
        'S' => vec![Dot, Dot, Dot],
        'T' => vec![Dash],
        'U' => vec![Dot, Dot, Dash],
        'V' => vec![Dot, Dot, Dot, Dash],
        'W' => vec![Dot, Dash, Dash],
        'X' => vec![Dash, Dot, Dot, Dash],
        'Y' => vec![Dash, Dot, Dash, Dash],
        'Z' => vec![Dash, Dash, Dot, Dot],
        '0' => vec![Dash, Dash, Dash, Dash, Dash],
        '1' => vec![Dot, Dash, Dash, Dash, Dash],
        '2' => vec![Dot, Dot, Dash, Dash, Dash],
        '3' => vec![Dot, Dot, Dot, Dash, Dash],
        '4' => vec![Dot, Dot, Dot, Dot, Dash],
        '5' => vec![Dot, Dot, Dot, Dot, Dot],
        '6' => vec![Dash, Dot, Dot, Dot, Dot],
        '7' => vec![Dash, Dash, Dot, Dot, Dot],
        '8' => vec![Dash, Dash, Dash, Dot, Dot],
        '9' => vec![Dash, Dash, Dash, Dash, Dot],
        _ => vec![],
    }
}

/// Maps a sequence of Morse symbols back to a character.
fn morse_to_char(symbols: &[MorseSymbol]) -> char {
    use MorseSymbol::{Dash, Dot};
    match symbols {
        [Dot, Dash] => 'A',
        [Dash, Dot, Dot, Dot] => 'B',
        [Dash, Dot, Dash, Dot] => 'C',
        [Dash, Dot, Dot] => 'D',
        [Dot] => 'E',
        [Dot, Dot, Dash, Dot] => 'F',
        [Dash, Dash, Dot] => 'G',
        [Dot, Dot, Dot, Dot] => 'H',
        [Dot, Dot] => 'I',
        [Dot, Dash, Dash, Dash] => 'J',
        [Dash, Dot, Dash] => 'K',
        [Dot, Dash, Dot, Dot] => 'L',
        [Dash, Dash] => 'M',
        [Dash, Dot] => 'N',
        [Dash, Dash, Dash] => 'O',
        [Dot, Dash, Dash, Dot] => 'P',
        [Dash, Dash, Dot, Dash] => 'Q',
        [Dot, Dash, Dot] => 'R',
        [Dot, Dot, Dot] => 'S',
        [Dash] => 'T',
        [Dot, Dot, Dash] => 'U',
        [Dot, Dot, Dot, Dash] => 'V',
        [Dot, Dash, Dash] => 'W',
        [Dash, Dot, Dot, Dash] => 'X',
        [Dash, Dot, Dash, Dash] => 'Y',
        [Dash, Dash, Dot, Dot] => 'Z',
        [Dash, Dash, Dash, Dash, Dash] => '0',
        [Dot, Dash, Dash, Dash, Dash] => '1',
        [Dot, Dot, Dash, Dash, Dash] => '2',
        [Dot, Dot, Dot, Dash, Dash] => '3',
        [Dot, Dot, Dot, Dot, Dash] => '4',
        [Dot, Dot, Dot, Dot, Dot] => '5',
        [Dash, Dot, Dot, Dot, Dot] => '6',
        [Dash, Dash, Dot, Dot, Dot] => '7',
        [Dash, Dash, Dash, Dot, Dot] => '8',
        [Dash, Dash, Dash, Dash, Dot] => '9',
        _ => '?',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_sos() {
        let signal = encode_morse("SOS");
        // S = ... (3 dots), O = --- (3 dashes), S = ... (3 dots)
        let freq_parts: Vec<f32> = signal.iter().map(|&(f, _)| f).collect();
        assert!(freq_parts.contains(&MORSE_FREQ));
        assert!(freq_parts.contains(&0.0)); // gaps exist
    }

    #[test]
    fn test_char_to_morse_s() {
        let s = char_to_morse('S');
        assert_eq!(
            s,
            vec![MorseSymbol::Dot, MorseSymbol::Dot, MorseSymbol::Dot]
        );
    }

    #[test]
    fn test_roundtrip_char() {
        for ch in 'A'..='Z' {
            let morse = char_to_morse(ch);
            let decoded = morse_to_char(&morse);
            assert_eq!(decoded, ch, "Failed roundtrip for {ch}");
        }
    }
}
