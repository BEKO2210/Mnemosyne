// SPDX-License-Identifier: AGPL-3.0-only
use core::mem;

/// Memory accounting for a single component.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryStats {
    /// Bytes occupied by the struct itself (`size_of::<T>()`).
    pub static_bytes: usize,
    /// Bytes occupied by dynamic buffers owned by the struct.
    pub dynamic_bytes: usize,
}

impl MemoryStats {
    /// Total memory estimate in bytes.
    #[inline]
    pub const fn total_bytes(self) -> usize {
        self.static_bytes + self.dynamic_bytes
    }
}

/// Target upper bound for default embedded RAM planning.
pub const EMBEDDED_RAM_LIMIT_BYTES: usize = 64 * 1024;
/// Size of one ring-buffer sample.
pub const SAMPLE_BYTES: usize = mem::size_of::<f32>();
