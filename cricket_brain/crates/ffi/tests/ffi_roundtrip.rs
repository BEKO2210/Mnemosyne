// SPDX-License-Identifier: AGPL-3.0-only
//! Integration tests for the C-compatible FFI layer.
//!
//! These tests call the `extern "C"` functions directly from Rust to verify
//! the contract that C/C++/Swift consumers rely on.

use cricket_brain::error_codes::*;
use cricket_brain_ffi::*;
use std::ptr;

#[test]
fn brain_new_returns_ok_with_valid_params() {
    let mut handle: *mut BrainHandle = ptr::null_mut();
    let rc = unsafe { brain_new(&mut handle, 5, 4000.0, 5000.0) };
    assert_eq!(rc, CRICKET_OK);
    assert!(!handle.is_null());
    unsafe { brain_free(handle) };
}

#[test]
fn brain_new_null_out_handle_returns_err_null() {
    let rc = unsafe { brain_new(ptr::null_mut(), 5, 4000.0, 5000.0) };
    assert_eq!(rc, CRICKET_ERR_NULL);
}

#[test]
fn brain_step_produces_output() {
    let mut handle: *mut BrainHandle = ptr::null_mut();
    let rc = unsafe { brain_new(&mut handle, 5, 4000.0, 5000.0) };
    assert_eq!(rc, CRICKET_OK);

    let mut output: f32 = -1.0;
    // Feed a signal in the resonant band for several steps
    for _ in 0..100 {
        let step_rc = unsafe { brain_step(handle, 4500.0, &mut output) };
        assert_eq!(step_rc, CRICKET_OK);
    }
    // Output should be a valid f32 (not NaN or the initial sentinel)
    assert!(output.is_finite());

    unsafe { brain_free(handle) };
}

#[test]
fn brain_step_null_handle_returns_err_null() {
    let mut output: f32 = 0.0;
    let rc = unsafe { brain_step(ptr::null_mut(), 4500.0, &mut output) };
    assert_eq!(rc, CRICKET_ERR_NULL);
}

#[test]
fn brain_step_null_output_returns_err_null() {
    let mut handle: *mut BrainHandle = ptr::null_mut();
    unsafe { brain_new(&mut handle, 5, 4000.0, 5000.0) };

    let rc = unsafe { brain_step(handle, 4500.0, ptr::null_mut()) };
    assert_eq!(rc, CRICKET_ERR_NULL);

    unsafe { brain_free(handle) };
}

#[test]
fn brain_step_negative_freq_returns_err_invalid_input() {
    let mut handle: *mut BrainHandle = ptr::null_mut();
    unsafe { brain_new(&mut handle, 5, 4000.0, 5000.0) };

    let mut output: f32 = 0.0;
    let rc = unsafe { brain_step(handle, -100.0, &mut output) };
    assert_eq!(rc, CRICKET_ERR_INVALID_INPUT);

    unsafe { brain_free(handle) };
}

#[test]
fn brain_step_nan_freq_returns_err_invalid_input() {
    let mut handle: *mut BrainHandle = ptr::null_mut();
    unsafe { brain_new(&mut handle, 5, 4000.0, 5000.0) };

    let mut output: f32 = 0.0;
    let rc = unsafe { brain_step(handle, f32::NAN, &mut output) };
    assert_eq!(rc, CRICKET_ERR_INVALID_INPUT);

    unsafe { brain_free(handle) };
}

#[test]
fn brain_get_status_returns_valid_state() {
    let mut handle: *mut BrainHandle = ptr::null_mut();
    unsafe { brain_new(&mut handle, 5, 4000.0, 5000.0) };

    let mut output: f32 = 0.0;
    for _ in 0..10 {
        unsafe { brain_step(handle, 4500.0, &mut output) };
    }

    let mut status = BrainStatus {
        time_step: 0,
        last_output: 0.0,
        global_sensitivity: 0.0,
        last_error_code: -1,
    };
    let rc = unsafe { brain_get_status(handle, &mut status) };
    assert_eq!(rc, CRICKET_OK);
    assert_eq!(status.time_step, 10);
    assert!(status.global_sensitivity > 0.0);
    assert_eq!(status.last_error_code, CRICKET_OK);

    unsafe { brain_free(handle) };
}

#[test]
fn brain_get_status_null_returns_err_null() {
    let mut status = BrainStatus {
        time_step: 0,
        last_output: 0.0,
        global_sensitivity: 0.0,
        last_error_code: 0,
    };
    let rc = unsafe { brain_get_status(ptr::null(), &mut status) };
    assert_eq!(rc, CRICKET_ERR_NULL);
}

#[test]
fn brain_free_null_is_safe() {
    // Must not crash or double-free
    unsafe { brain_free(ptr::null_mut()) };
}

#[test]
fn brain_get_version_returns_non_null() {
    let ptr = brain_get_version();
    assert!(!ptr.is_null());
    let version = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let s = version.to_str().expect("valid utf-8 version string");
    assert!(
        s.contains("3.0.0"),
        "version should contain 3.0.0, got: {s}"
    );
}

#[test]
fn error_code_helpers_match_constants() {
    assert_eq!(cricket_error_ok(), CRICKET_OK);
    assert_eq!(cricket_error_null(), CRICKET_ERR_NULL);
    assert_eq!(cricket_error_invalid_config(), CRICKET_ERR_INVALID_CONFIG);
    assert_eq!(cricket_error_token_not_found(), CRICKET_ERR_TOKEN_NOT_FOUND);
    assert_eq!(cricket_error_invalid_input(), CRICKET_ERR_INVALID_INPUT);
    assert_eq!(cricket_error_internal(), CRICKET_ERR_INTERNAL);
}
