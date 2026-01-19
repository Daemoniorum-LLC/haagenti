/**
 * HCT Reference Decoder Test Program
 *
 * Validates the C decoder against the HCT specification test vectors.
 *
 * Build:
 *   gcc -O2 -o test_decoder test_decoder.c -lm
 *
 * Run:
 *   ./test_decoder
 */

#define HCT_IMPLEMENTATION
#include "hct_decoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * Test Vectors from HCT Specification Section 7
 * ============================================================================ */

/* Test Vector 1: Sequential 4x4 */
static float TV_SEQUENTIAL_4X4[] = {
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f
};

/* Test Vector 2: Identity 4x4 */
static float TV_IDENTITY_4X4[] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

/* Test Vector 3: Constant 4x4 (all 42.0) */
static float TV_CONSTANT_4X4[] = {
    42.0f, 42.0f, 42.0f, 42.0f,
    42.0f, 42.0f, 42.0f, 42.0f,
    42.0f, 42.0f, 42.0f, 42.0f,
    42.0f, 42.0f, 42.0f, 42.0f
};

/* Test Vector 4: Zeros 4x4 */
static float TV_ZEROS_4X4[] = {
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f
};

/* ============================================================================
 * Test Functions
 * ============================================================================ */

/**
 * Test DCT/IDCT roundtrip accuracy.
 */
int test_dct_roundtrip(const char* name, const float* input, size_t rows, size_t cols) {
    size_t total = rows * cols;

    float* dct = (float*)malloc(total * sizeof(float));
    float* reconstructed = (float*)malloc(total * sizeof(float));
    if (!dct || !reconstructed) {
        printf("FAIL: %s - memory allocation failed\n", name);
        free(dct);
        free(reconstructed);
        return 1;
    }

    /* Forward DCT */
    hct_dct_2d(input, dct, rows, cols);

    /* Inverse DCT */
    hct_idct_2d(dct, reconstructed, rows, cols);

    /* Check cosine similarity */
    float cosine = hct_cosine_similarity(input, reconstructed, total);

    /* For non-zero inputs, expect near-perfect roundtrip */
    int is_zero = 1;
    for (size_t i = 0; i < total; i++) {
        if (input[i] != 0.0f) {
            is_zero = 0;
            break;
        }
    }

    int pass;
    if (is_zero) {
        /* Zero input: just check output is also zero */
        float max_err = 0.0f;
        for (size_t i = 0; i < total; i++) {
            float err = fabsf(reconstructed[i]);
            if (err > max_err) max_err = err;
        }
        pass = (max_err < 1e-6f);
    } else {
        pass = (cosine > 0.999999f);
    }

    if (pass) {
        printf("PASS: %s (cosine=%.6f)\n", name, cosine);
    } else {
        printf("FAIL: %s (cosine=%.6f, expected >0.999999)\n", name, cosine);
    }

    free(dct);
    free(reconstructed);
    return pass ? 0 : 1;
}

/**
 * Test truncated DCT reconstruction (simulates compression).
 */
int test_truncated_reconstruction(const char* name, const float* input,
                                   size_t rows, size_t cols,
                                   float retention, float min_cosine) {
    size_t total = rows * cols;
    size_t keep = (size_t)(total * retention + 0.5f);

    float* dct = (float*)malloc(total * sizeof(float));
    float* truncated = (float*)calloc(total, sizeof(float));
    float* reconstructed = (float*)malloc(total * sizeof(float));
    if (!dct || !truncated || !reconstructed) {
        printf("FAIL: %s - memory allocation failed\n", name);
        free(dct);
        free(truncated);
        free(reconstructed);
        return 1;
    }

    /* Forward DCT */
    hct_dct_2d(input, dct, rows, cols);

    /* Sort by magnitude and keep top 'keep' coefficients */
    /* Simple O(n*k) selection for small test cases */
    for (size_t k = 0; k < keep; k++) {
        size_t max_idx = 0;
        float max_val = 0.0f;

        for (size_t i = 0; i < total; i++) {
            float abs_val = fabsf(dct[i]);
            if (abs_val > max_val) {
                max_val = abs_val;
                max_idx = i;
            }
        }

        truncated[max_idx] = dct[max_idx];
        dct[max_idx] = 0.0f; /* Mark as used */
    }

    /* Inverse DCT */
    hct_idct_2d(truncated, reconstructed, rows, cols);

    /* Check cosine similarity */
    float cosine = hct_cosine_similarity(input, reconstructed, total);

    int pass = (cosine >= min_cosine);
    if (pass) {
        printf("PASS: %s @ %.0f%% retention (cosine=%.6f >= %.3f)\n",
               name, retention * 100.0f, cosine, min_cosine);
    } else {
        printf("FAIL: %s @ %.0f%% retention (cosine=%.6f < %.3f)\n",
               name, retention * 100.0f, cosine, min_cosine);
    }

    free(dct);
    free(truncated);
    free(reconstructed);
    return pass ? 0 : 1;
}

/**
 * Test DC-only compression for constant input.
 */
int test_constant_dc_only(void) {
    size_t rows = 4, cols = 4, total = 16;

    float* dct = (float*)malloc(total * sizeof(float));
    float* reconstructed = (float*)malloc(total * sizeof(float));
    if (!dct || !reconstructed) {
        printf("FAIL: constant_dc_only - memory allocation failed\n");
        free(dct);
        free(reconstructed);
        return 1;
    }

    /* DCT of constant input */
    hct_dct_2d(TV_CONSTANT_4X4, dct, rows, cols);

    /* Verify DC coefficient is 168.0 (42 * 4 = 168 with normalization) */
    float dc = dct[0];
    float expected_dc = 168.0f;

    /* Keep only DC */
    float* dc_only = (float*)calloc(total, sizeof(float));
    dc_only[0] = dct[0];

    /* Reconstruct */
    hct_idct_2d(dc_only, reconstructed, rows, cols);

    float cosine = hct_cosine_similarity(TV_CONSTANT_4X4, reconstructed, total);

    int pass = (cosine > 0.9999f) && (fabsf(dc - expected_dc) < 0.01f);
    if (pass) {
        printf("PASS: constant_dc_only (DC=%.1f, cosine=%.6f)\n", dc, cosine);
    } else {
        printf("FAIL: constant_dc_only (DC=%.1f expected %.1f, cosine=%.6f)\n",
               dc, expected_dc, cosine);
    }

    free(dct);
    free(dc_only);
    free(reconstructed);
    return pass ? 0 : 1;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    int failures = 0;

    printf("HCT Reference Decoder Test Suite\n");
    printf("=================================\n\n");

    printf("DCT/IDCT Roundtrip Tests:\n");
    printf("-------------------------\n");
    failures += test_dct_roundtrip("sequential_4x4", TV_SEQUENTIAL_4X4, 4, 4);
    failures += test_dct_roundtrip("identity_4x4", TV_IDENTITY_4X4, 4, 4);
    failures += test_dct_roundtrip("constant_4x4", TV_CONSTANT_4X4, 4, 4);
    failures += test_dct_roundtrip("zeros_4x4", TV_ZEROS_4X4, 4, 4);
    printf("\n");

    printf("Truncated Reconstruction Tests:\n");
    printf("-------------------------------\n");
    failures += test_truncated_reconstruction("sequential_4x4", TV_SEQUENTIAL_4X4, 4, 4, 0.5f, 0.99f);
    failures += test_truncated_reconstruction("identity_4x4", TV_IDENTITY_4X4, 4, 4, 0.5f, 0.99f);
    failures += test_truncated_reconstruction("constant_4x4", TV_CONSTANT_4X4, 4, 4, 0.25f, 0.99f);
    printf("\n");

    printf("DC-Only Test:\n");
    printf("-------------\n");
    failures += test_constant_dc_only();
    printf("\n");

    printf("=================================\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("%d test(s) FAILED\n", failures);
        return 1;
    }
}
