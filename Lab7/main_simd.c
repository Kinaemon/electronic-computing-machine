#include "common.h"
#include <stdio.h>
#include <stdlib.h>

static void matmul_scalar(const float* restrict A, const float* restrict B, float* restrict C, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

static void matadd_scalar(float* restrict A, const float* restrict B, int N) {
    int total = N * N;
    for (int i = 0; i < total; ++i) {
        A[i] += B[i];
    }
}

static void invert_scalar(const float* A, float* A_inv, int N, int M) {
    float norm1 = 0.0f, norm_inf = 0.0f;
    compute_norms(A, N, &norm1, &norm_inf);
    float scale = 1.0f / (norm1 * norm_inf);

    float* B      = alloc_matrix(N);
    float* R      = alloc_matrix(N);
    float* S      = alloc_matrix(N);
    float* R_pow  = alloc_matrix(N);
    float* R_next = alloc_matrix(N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i * N + j] = A[j * N + i] * scale;
        }
    }

    // R = I - B * A
    matmul_scalar(B, A, R, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            R[i * N + j] = (i == j ? 1.0f : 0.0f) - R[i * N + j];
        }
    }

    set_identity(S, N);
    set_identity(R_pow, N);

    // S = I + R + R^2 + ... + R^(M-1)
    for (int m = 1; m < M; ++m) {
        matmul_scalar(R_pow, R, R_next, N);
        float* tmp = R_pow;
        R_pow = R_next;
        R_next = tmp;

        matadd_scalar(S, R_pow, N);
    }

    matmul_scalar(S, B, A_inv, N);

    free(B);
    free(R);
    free(S);
    free(R_pow);
    free(R_next);
}

int main(int argc, char** argv) {
    int N = 0, M = 0;
    float* A = NULL;
    int interactive = (argc < 3);

    if (interactive) {
        if (read_input(&N, &M, &A) != 0) return EXIT_FAILURE;
    } else {
        N = atoi(argv[1]);
        M = atoi(argv[2]);
        A = alloc_matrix(N);
        generate_test_matrix(A, N);
    }

    float* A_inv = alloc_matrix(N);

    double t1 = get_time_sec();
    invert_scalar(A, A_inv, N, M);
    double t2 = get_time_sec();

    if (interactive && N <= 16) {
        print_matrix("Обратная матрица A^-1 (Scalar)", A_inv, N);
    }

    printf("[Scalar] N = %d, M = %d | Time: %.4f s", N, M, t2 - t1);
    if (N <= 512) {
        printf(" | Residual ||A*A^-1 - I||_inf: %e", compute_residual(A, A_inv, N));
    }
    printf("\n");

    free(A);
    free(A_inv);
    return EXIT_SUCCESS;
}