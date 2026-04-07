// SPDX-License-Identifier: AGPL-3.0-only
use cricket_brain::brain::{BrainConfig, CricketBrain};
use cricket_brain::error::CricketError;
use cricket_brain::error_codes::{
    CRICKET_ERR_INTERNAL, CRICKET_ERR_INVALID_CONFIG, CRICKET_ERR_INVALID_INPUT, CRICKET_ERR_NULL,
    CRICKET_ERR_TOKEN_NOT_FOUND, CRICKET_OK,
};

#[repr(C)]
pub struct BrainStatus {
    pub time_step: u64,
    pub last_output: f32,
    pub global_sensitivity: f32,
    pub last_error_code: i32,
}

pub struct BrainHandle {
    brain: CricketBrain,
    last_output: f32,
    last_error_code: i32,
}

fn error_code(err: &CricketError) -> i32 {
    match err {
        CricketError::InvalidConfiguration(_) => CRICKET_ERR_INVALID_CONFIG,
        CricketError::TokenNotFound(_) => CRICKET_ERR_TOKEN_NOT_FOUND,
        CricketError::InvalidInput(_) => CRICKET_ERR_INVALID_INPUT,
    }
}

/// Creates a new brain instance and returns it through `out_handle`.
///
/// # Safety
/// - `out_handle` must be a valid, writable pointer to receive the allocated handle.
/// - Caller must later release the returned handle using [`brain_free`].
#[no_mangle]
pub unsafe extern "C" fn brain_new(
    out_handle: *mut *mut BrainHandle,
    n_neurons: usize,
    min_freq: f32,
    max_freq: f32,
) -> i32 {
    if out_handle.is_null() {
        return CRICKET_ERR_NULL;
    }

    let config = BrainConfig::default()
        .with_neurons(n_neurons)
        .with_freq_range(min_freq, max_freq);
    let brain = match CricketBrain::new(config) {
        Ok(brain) => brain,
        Err(err) => return error_code(&err),
    };

    let boxed = Box::new(BrainHandle {
        brain,
        last_output: 0.0,
        last_error_code: CRICKET_OK,
    });
    // SAFETY: `out_handle` is checked non-null and points to caller-owned storage.
    *out_handle = Box::into_raw(boxed);
    CRICKET_OK
}

/// Steps the brain for one sample and writes output to `out_output`.
///
/// # Safety
/// - `handle` must be a valid pointer returned by [`brain_new`] and not yet freed.
/// - `out_output` must be a valid, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn brain_step(
    handle: *mut BrainHandle,
    input_freq: f32,
    out_output: *mut f32,
) -> i32 {
    if handle.is_null() || out_output.is_null() {
        return CRICKET_ERR_NULL;
    }
    if !input_freq.is_finite() || input_freq < 0.0 {
        return CRICKET_ERR_INVALID_INPUT;
    }

    // SAFETY: pointers validated above.
    let handle_ref = &mut *handle;
    let output = handle_ref.brain.step(input_freq);
    handle_ref.last_output = output;
    handle_ref.last_error_code = CRICKET_OK;
    // SAFETY: pointer validated above.
    *out_output = output;
    CRICKET_OK
}

/// Reads lightweight status fields from the brain.
///
/// # Safety
/// - `handle` must be a valid pointer returned by [`brain_new`] and not yet freed.
/// - `out_status` must be a valid, writable pointer.
#[no_mangle]
pub unsafe extern "C" fn brain_get_status(
    handle: *const BrainHandle,
    out_status: *mut BrainStatus,
) -> i32 {
    if handle.is_null() || out_status.is_null() {
        return CRICKET_ERR_NULL;
    }
    // SAFETY: pointers validated above.
    let handle_ref = &*handle;
    // SAFETY: pointer validated above.
    *out_status = BrainStatus {
        time_step: handle_ref.brain.time_step as u64,
        last_output: handle_ref.last_output,
        global_sensitivity: handle_ref.brain.global_sensitivity,
        last_error_code: handle_ref.last_error_code,
    };
    CRICKET_OK
}

/// Frees a brain instance allocated by `brain_new`.
///
/// # Safety
/// - `handle` must be either null or a pointer previously returned by [`brain_new`].
/// - The pointer must not be used after this call.
#[no_mangle]
pub unsafe extern "C" fn brain_free(handle: *mut BrainHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: pointer was allocated via Box::into_raw in brain_new.
    drop(Box::from_raw(handle));
}

#[no_mangle]
pub extern "C" fn cricket_error_ok() -> i32 {
    CRICKET_OK
}

#[no_mangle]
pub extern "C" fn cricket_error_internal() -> i32 {
    CRICKET_ERR_INTERNAL
}

#[no_mangle]
pub extern "C" fn cricket_error_null() -> i32 {
    CRICKET_ERR_NULL
}

#[no_mangle]
pub extern "C" fn cricket_error_invalid_config() -> i32 {
    CRICKET_ERR_INVALID_CONFIG
}

#[no_mangle]
pub extern "C" fn cricket_error_token_not_found() -> i32 {
    CRICKET_ERR_TOKEN_NOT_FOUND
}

#[no_mangle]
pub extern "C" fn cricket_error_invalid_input() -> i32 {
    CRICKET_ERR_INVALID_INPUT
}

#[no_mangle]
pub extern "C" fn brain_get_version() -> *const core::ffi::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const core::ffi::c_char
}
