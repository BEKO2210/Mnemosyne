// SPDX-License-Identifier: AGPL-3.0-only
use alloc::string::String;
use core::fmt;

/// Central error type for public Cricket-Brain APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CricketError {
    /// Invalid runtime or construction configuration.
    InvalidConfiguration(String),
    /// Referenced token does not exist in the vocabulary.
    TokenNotFound(String),
    /// Generic invalid input for API calls.
    InvalidInput(String),
}

impl fmt::Display for CricketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CricketError::InvalidConfiguration(msg) => {
                write!(f, "invalid configuration: {msg}")
            }
            CricketError::TokenNotFound(token) => {
                write!(f, "token not found in vocabulary: '{token}'")
            }
            CricketError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}
