/**
 * HCT Reference Decoder - C Implementation
 *
 * This is the reference C implementation of HCT (Holographic Compressed Tensor)
 * decompression, suitable for embedding in inference runtimes.
 *
 * Features:
 * - Single-header implementation (include hct_decoder.h)
 * - No external dependencies (stdlib only)
 * - Matches Rust reference implementation exactly
 *
 * Usage:
 *   #define HCT_IMPLEMENTATION
 *   #include "hct_decoder.h"
 *
 *   float* tensor = hct_decode(compressed_data, data_len, rows, cols);
 *   // ... use tensor ...
 *   free(tensor);
 *
 * License: MIT (same as Haagenti)
 */

#ifndef HCT_DECODER_H
#define HCT_DECODER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

/**
 * Decode an HCT-compressed tensor.
 *
 * @param data       Pointer to compressed data (HCT V3 format)
 * @param data_len   Length of compressed data in bytes
 * @param rows       Number of rows in output tensor
 * @param cols       Number of columns in output tensor
 * @return           Newly allocated float array (rows*cols elements), or NULL on error
 *
 * Caller is responsible for freeing the returned array.
 */
float* hct_decode(const uint8_t* data, size_t data_len, size_t rows, size_t cols);

/**
 * Compute 2D DCT-II (forward transform).
 *
 * @param input      Input array (rows*cols floats, row-major)
 * @param output     Output array (rows*cols floats, row-major)
 * @param rows       Number of rows
 * @param cols       Number of columns
 */
void hct_dct_2d(const float* input, float* output, size_t rows, size_t cols);

/**
 * Compute 2D IDCT (inverse DCT, DCT-III).
 *
 * @param input      Input DCT coefficients (rows*cols floats, row-major)
 * @param output     Output array (rows*cols floats, row-major)
 * @param rows       Number of rows
 * @param cols       Number of columns
 */
void hct_idct_2d(const float* input, float* output, size_t rows, size_t cols);

/**
 * Compute cosine similarity between two arrays.
 *
 * @param a          First array
 * @param b          Second array
 * @param len        Number of elements
 * @return           Cosine similarity in range [-1, 1], or 0 if either array has zero norm
 */
float hct_cosine_similarity(const float* a, const float* b, size_t len);

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * IMPLEMENTATION
 * ============================================================================ */

#ifdef HCT_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --------------------------------------------------------------------------
 * Internal: 1D DCT-II
 * -------------------------------------------------------------------------- */
static void hct_dct_1d(const float* input, float* output, size_t n) {
    float scale = sqrtf(2.0f / (float)n);
    float scale0 = scale * 0.7071067811865476f; /* 1/sqrt(2) */

    for (size_t k = 0; k < n; k++) {
        float sum = 0.0f;
        float pi_2n = (float)M_PI / (2.0f * (float)n);

        for (size_t i = 0; i < n; i++) {
            float angle = (2.0f * (float)i + 1.0f) * (float)k * pi_2n;
            sum += input[i] * cosf(angle);
        }

        output[k] = sum * (k == 0 ? scale0 : scale);
    }
}

/* --------------------------------------------------------------------------
 * Internal: 1D IDCT (DCT-III)
 * -------------------------------------------------------------------------- */
static void hct_idct_1d(const float* input, float* output, size_t n) {
    float scale = sqrtf(2.0f / (float)n);
    float pi_2n = (float)M_PI / (2.0f * (float)n);

    for (size_t i = 0; i < n; i++) {
        /* DC term (k=0) */
        float sum = input[0] * scale * 0.7071067811865476f;

        /* AC terms (k>0) */
        for (size_t k = 1; k < n; k++) {
            float angle = (2.0f * (float)i + 1.0f) * (float)k * pi_2n;
            sum += input[k] * scale * cosf(angle);
        }

        output[i] = sum;
    }
}

/* --------------------------------------------------------------------------
 * Public: 2D DCT-II (separable: rows then columns)
 * -------------------------------------------------------------------------- */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
void hct_dct_2d(const float* input, float* output, size_t rows, size_t cols) {
    /* Allocate all buffers upfront */
    float* temp = NULL;
    float* col_in = NULL;
    float* col_out = NULL;

    temp = (float*)malloc(rows * cols * sizeof(float));
    col_in = (float*)malloc(rows * sizeof(float));
    col_out = (float*)malloc(rows * sizeof(float));

    if (!temp || !col_in || !col_out) {
        free(temp);
        free(col_in);
        free(col_out);
        return;
    }

    /* DCT on rows */
    for (size_t r = 0; r < rows; r++) {
        hct_dct_1d(&input[r * cols], &temp[r * cols], cols);
    }

    /* DCT on columns */
    for (size_t c = 0; c < cols; c++) {
        /* Extract column */
        for (size_t r = 0; r < rows; r++) {
            col_in[r] = temp[r * cols + c];
        }

        /* DCT on column */
        hct_dct_1d(col_in, col_out, rows);

        /* Store result */
        for (size_t r = 0; r < rows; r++) {
            output[r * cols + c] = col_out[r];
        }
    }

    free(temp);
    free(col_in);
    free(col_out);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

/* --------------------------------------------------------------------------
 * Public: 2D IDCT (separable: columns then rows)
 * -------------------------------------------------------------------------- */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
void hct_idct_2d(const float* input, float* output, size_t rows, size_t cols) {
    /* Allocate all buffers upfront */
    float* temp = NULL;
    float* col_in = NULL;
    float* col_out = NULL;

    temp = (float*)malloc(rows * cols * sizeof(float));
    col_in = (float*)malloc(rows * sizeof(float));
    col_out = (float*)malloc(rows * sizeof(float));

    if (!temp || !col_in || !col_out) {
        free(temp);
        free(col_in);
        free(col_out);
        return;
    }

    /* IDCT on columns first */
    for (size_t c = 0; c < cols; c++) {
        /* Extract column */
        for (size_t r = 0; r < rows; r++) {
            col_in[r] = input[r * cols + c];
        }

        /* IDCT on column */
        hct_idct_1d(col_in, col_out, rows);

        /* Store result */
        for (size_t r = 0; r < rows; r++) {
            temp[r * cols + c] = col_out[r];
        }
    }

    /* IDCT on rows */
    for (size_t r = 0; r < rows; r++) {
        hct_idct_1d(&temp[r * cols], &output[r * cols], cols);
    }

    free(temp);
    free(col_in);
    free(col_out);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

/* --------------------------------------------------------------------------
 * Public: Cosine similarity
 * -------------------------------------------------------------------------- */
float hct_cosine_similarity(const float* a, const float* b, size_t len) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < len; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return (norm_a == 0.0f && norm_b == 0.0f) ? 1.0f : 0.0f;
    }

    return dot / (norm_a * norm_b);
}

/* --------------------------------------------------------------------------
 * Internal: Read little-endian integers
 * -------------------------------------------------------------------------- */
static uint16_t read_u16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* --------------------------------------------------------------------------
 * Internal: Convert FP16 to FP32
 * -------------------------------------------------------------------------- */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        /* Subnormal or zero */
        if (mant == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, sizeof(f));
            return f;
        }
        /* Subnormal: normalize */
        exp = 1;
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
        exp = exp - 15 + 127;
    } else if (exp == 31) {
        /* Inf or NaN */
        exp = 255;
    } else {
        /* Normal: adjust exponent bias */
        exp = exp - 15 + 127;
    }

    uint32_t result = sign | (exp << 23) | (mant << 13);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

/* --------------------------------------------------------------------------
 * Public: Decode HCT V3 compressed tensor
 * -------------------------------------------------------------------------- */
float* hct_decode(const uint8_t* data, size_t data_len, size_t rows, size_t cols) {
    if (!data || data_len < 2) return NULL;

    size_t total = rows * cols;
    if (total == 0) return NULL;

    /* Allocate output (sparse DCT coefficients) */
    float* dct = (float*)calloc(total, sizeof(float));
    if (!dct) return NULL;

    /* Read header: num_fragments (u16 LE) */
    uint16_t num_fragments = read_u16_le(data);
    size_t pos = 2;

    /* Process each fragment */
    for (uint16_t frag = 0; frag < num_fragments && pos < data_len; frag++) {
        if (pos + 16 > data_len) break; /* Need at least header */

        /* Fragment header */
        /* uint16_t frag_index = read_u16_le(data + pos); */ pos += 2;
        /* uint16_t flags = read_u16_le(data + pos); */ pos += 2;
        /* uint64_t checksum = ... */ pos += 8;
        uint32_t frag_data_len = read_u32_le(data + pos); pos += 4;

        if (pos + frag_data_len > data_len) break;

        const uint8_t* frag_data = data + pos;
        pos += frag_data_len;

        /* Parse fragment data (V3 with bitmap) */
        if (frag_data_len < 4) continue;

        uint32_t num_coefficients = read_u32_le(frag_data);
        size_t frag_pos = 4;

        /* Bitmap: (total + 7) / 8 bytes */
        size_t bitmap_bytes = (total + 7) / 8;
        if (frag_pos + bitmap_bytes > frag_data_len) continue;

        const uint8_t* bitmap = frag_data + frag_pos;
        frag_pos += bitmap_bytes;

        /* Coefficients: num_coefficients * 2 bytes (FP16) */
        if (frag_pos + num_coefficients * 2 > frag_data_len) continue;

        const uint8_t* coeffs = frag_data + frag_pos;

        /* Scatter coefficients using bitmap */
        size_t coeff_idx = 0;
        for (size_t i = 0; i < total && coeff_idx < num_coefficients; i++) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;

            if (bitmap[byte_idx] & (1 << bit_idx)) {
                uint16_t fp16 = read_u16_le(coeffs + coeff_idx * 2);
                dct[i] = fp16_to_fp32(fp16);
                coeff_idx++;
            }
        }
    }

    /* Apply IDCT to reconstruct tensor */
    float* output = (float*)malloc(total * sizeof(float));
    if (!output) {
        free(dct);
        return NULL;
    }

    hct_idct_2d(dct, output, rows, cols);

    free(dct);
    return output;
}

#endif /* HCT_IMPLEMENTATION */

#endif /* HCT_DECODER_H */
