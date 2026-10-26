#include "observer_d_neon.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define OBSERVER_D_HAS_NEON 1
#else
#define OBSERVER_D_HAS_NEON 0
#endif

static inline float dot_f32(const float *left, const float *right, size_t length) {
    size_t index = 0;
    float result = 0.0f;
#if OBSERVER_D_HAS_NEON
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    for (; index + 4 <= length; index += 4) {
        accumulator = vfmaq_f32(
            accumulator,
            vld1q_f32(left + index),
            vld1q_f32(right + index));
    }
    result = vaddvq_f32(accumulator);
#endif
    for (; index < length; ++index) {
        result += left[index] * right[index];
    }
    return result;
}

static inline float gelu_tanh(float value) {
    const float scale = 0.7978845608028654f; /* sqrt(2 / pi) */
    const float cubic = 0.044715f * value * value * value;
    return 0.5f * value * (1.0f + tanhf(scale * (value + cubic)));
}

int observer_d_encode(
    const float *features,
    size_t rows,
    size_t input_dim,
    size_t hidden_dim,
    size_t latent_dim,
    const float *weight1,
    const float *bias1,
    const float *weight2,
    const float *bias2,
    float *latents) {
    if (!features || !weight1 || !bias1 || !weight2 || !bias2 || !latents ||
        rows == 0 || input_dim == 0 || hidden_dim == 0 || latent_dim == 0) {
        return 1;
    }

    float *hidden = (float *)malloc(hidden_dim * sizeof(float));
    if (!hidden) {
        return 2;
    }

    for (size_t row = 0; row < rows; ++row) {
        const float *input = features + row * input_dim;
        for (size_t output = 0; output < hidden_dim; ++output) {
            const float *weight = weight1 + output * input_dim;
            hidden[output] = gelu_tanh(dot_f32(input, weight, input_dim) + bias1[output]);
        }
        for (size_t output = 0; output < latent_dim; ++output) {
            const float *weight = weight2 + output * hidden_dim;
            latents[row * latent_dim + output] =
                dot_f32(hidden, weight, hidden_dim) + bias2[output];
        }
    }

    free(hidden);
    return 0;
}

const char *observer_d_backend(void) {
#if OBSERVER_D_HAS_NEON
    return "c-neon";
#else
    return "c-scalar";
#endif
}
