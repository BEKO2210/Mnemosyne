#pragma once

/* Generated with cbindgen:0.28.0 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct BrainHandle BrainHandle;

typedef struct BrainStatus {
  uint64_t time_step;
  float last_output;
  float global_sensitivity;
  int32_t last_error_code;
} BrainStatus;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Creates a new brain instance and returns it through `out_handle`.
 *
 * # Safety
 * - `out_handle` must be a valid, writable pointer to receive the allocated handle.
 * - Caller must later release the returned handle using [`brain_free`].
 */
int32_t brain_new(struct BrainHandle **out_handle,
                  uintptr_t n_neurons,
                  float min_freq,
                  float max_freq);

/**
 * Steps the brain for one sample and writes output to `out_output`.
 *
 * # Safety
 * - `handle` must be a valid pointer returned by [`brain_new`] and not yet freed.
 * - `out_output` must be a valid, writable pointer.
 */
int32_t brain_step(struct BrainHandle *handle, float input_freq, float *out_output);

/**
 * Reads lightweight status fields from the brain.
 *
 * # Safety
 * - `handle` must be a valid pointer returned by [`brain_new`] and not yet freed.
 * - `out_status` must be a valid, writable pointer.
 */
int32_t brain_get_status(const struct BrainHandle *handle, struct BrainStatus *out_status);

/**
 * Frees a brain instance allocated by `brain_new`.
 *
 * # Safety
 * - `handle` must be either null or a pointer previously returned by [`brain_new`].
 * - The pointer must not be used after this call.
 */
void brain_free(struct BrainHandle *handle);

int32_t cricket_error_ok(void);

int32_t cricket_error_internal(void);

int32_t cricket_error_null(void);

int32_t cricket_error_invalid_config(void);

int32_t cricket_error_token_not_found(void);

int32_t cricket_error_invalid_input(void);

const char *brain_get_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
