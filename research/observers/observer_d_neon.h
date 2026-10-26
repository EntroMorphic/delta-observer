#ifndef OBSERVER_D_NEON_H
#define OBSERVER_D_NEON_H

#include <stddef.h>

/*
 * Encode row-major features using
 *
 *   z = W2 * GELU(W1 * x + b1) + b2
 *
 * Weights use PyTorch's row-major [out_features, in_features] layout.
 * Returns zero on success.
 */
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
    float *latents);

const char *observer_d_backend(void);

#endif
