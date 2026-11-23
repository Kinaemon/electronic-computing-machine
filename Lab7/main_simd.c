#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

static void matmul_simd(const float* restrict A, const float* restrict B, float* restrict C, int N) {
    zero_matrix(C, N);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            __m256 v_a = _mm256_set1_ps(A[i * N + k]);
            int j = 0;

            // Развертка по 16 float (2 AVX-регистра по 8 float)
            for (; j <= N - 16; j += 16) {
                __m256 v_b0 = _mm256_loadu_ps(&B[k * N + j]);
                __m256 v_c0 = _mm256_loadu_ps(&C[i * N + j]);
                v_c0 = _mm256_fmadd_ps(v_a, v_b0, v_c0);
                _mm256_storeu_ps(&C[i * N + j], v_c0);

                __m256 v_b1 = _mm256_loadu_ps(&B[k * N + j + 8]);
                __m256 v_c1 = _mm256_loadu_ps(&C[i * N + j + 8]);
                v_c1 = _mm256_fmadd_ps(v_a, v_b1, v_c1);
                _mm256_storeu_ps(&C[i * N + j + 8], v_c1);
            }

            // Одиночный AVX-шаг (8 float)
            for (; j <= N - 8; j += 8) {
                __m256 v_b = _mm256_loadu_ps(&B[k * N + j]);
                __m256 v_c = _mm256_loadu_ps(&C[i * N + j]);
                v_c = _mm256_fmadd_ps(v_a, v_b, v_c);
                _mm256_storeu_ps(&C[i * N + j], v_c);
            }

            // Скалярный хвост для произвольного N (например, 3x3 из тестов)
            for (; j < N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

static void matadd_simd(float* restrict A, const float* restrict B, int N) {
    int total = N * N;
    int i = 0;
    for (; i <= total - 8; i += 8) {
        __m256 v_a = _mm256_loadu_ps(&A[i]);
        __m256 v_b = _mm256_loadu_ps(&B[i]);
        _mm256_storeu_ps(&A[i], _mm256_add_ps(v_a, v_b));
    }
    for (; i < total; ++i) {
        A[i] += B[i];
    }
}

static void invert_simd(const float* A, float* A_inv, int N, int M) {
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

    matmul_simd(B, A, R, N);

    int total = N * N;
    int idx = 0;
    __m256 v_zero = _mm256_setzero_ps();
    for (; idx <= total - 8; idx += 8) {
        __m256 v_r = _mm256_loadu_ps(&R[idx]);
        _mm256_storeu_ps(&R[idx], _mm256_sub_ps(v_zero, v_r));
    }
    for (; idx < total; ++idx) {
        R[idx] = -R[idx];
    }
    for (int i = 0; i < N; ++i) {
        R[i * N + i] += 1.0f;
    }

    set_identity(S, N);
    set_identity(R_pow, N);

    for (int m = 1; m < M; ++m) {
        matmul_simd(R_pow, R, R_next, N);
        float* tmp = R_pow;
        R_pow = R_next;
        R_next = tmp;

        matadd_simd(S, R_pow, N);
    }

    matmul_simd(S, B, A_inv, N);

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
    invert_simd(A, A_inv, N, M);
    double t2 = get_time_sec();

    if (interactive && N <= 16) {
        print_matrix("Обратная матрица A^-1 (SIMD AVX2)", A_inv, N);
    }

    float residual = compute_residual(A, A_inv, N);
    printf("[SIMD]   N = %d, M = %d | Time: %.4f s | Residual ||A*A^-1 - I||_inf: %e\n",
           N, M, t2 - t1, residual);

    free(A);
    free(A_inv);
    return EXIT_SUCCESS;
}
