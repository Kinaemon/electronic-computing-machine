#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <openblas/cblas.h>

static void invert_blas(const float* A, float* A_inv, int N, int M) {
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

    set_identity(R, N);

    // R = -1.0f * B * A + 1.0f * R (т.е. R = I - BA)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                -1.0f, B, N,
                A, N,
                1.0f, R, N);

    set_identity(S, N);
    set_identity(R_pow, N);

    for (int m = 1; m < M; ++m) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0f, R_pow, N,
                    R, N,
                    0.0f, R_next, N);
        float* tmp = R_pow;
        R_pow = R_next;
        R_next = tmp;

        cblas_saxpy(N * N, 1.0f, R_pow, 1, S, 1);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0f, S, N,
                B, N,
                0.0f, A_inv, N);

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
    invert_blas(A, A_inv, N, M);
    double t2 = get_time_sec();

    if (interactive && N <= 16) {
        print_matrix("Обратная матрица A^-1 (BLAS)", A_inv, N);
    }

    float residual = compute_residual(A, A_inv, N);
    printf("[BLAS]   N = %d, M = %d | Time: %.4f s | Residual ||A*A^-1 - I||_inf: %e\n",
           N, M, t2 - t1, residual);

    free(A);
    free(A_inv);
    return EXIT_SUCCESS;
}
