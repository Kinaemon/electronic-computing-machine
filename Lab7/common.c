#define _POSIX_C_SOURCE 199309L
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

float* alloc_matrix(int N) {
    size_t size = (size_t)N * (size_t)N * sizeof(float);
    size_t aligned_size = (size + 31) & ~((size_t)31);
    float* ptr = (float*)aligned_alloc(32, aligned_size);
    if (!ptr) {
        perror("aligned_alloc");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void zero_matrix(float* A, int N) {
    memset(A, 0, (size_t)N * (size_t)N * sizeof(float));
}

void set_identity(float* I, int N) {
    zero_matrix(I, N);
    for (int i = 0; i < N; ++i) {
        I[i * N + i] = 1.0f;
    }
}

void compute_norms(const float* A, int N, float* norm1, float* norm_inf) {
    *norm1 = 0.0f;
    *norm_inf = 0.0f;

    float* col_sums = (float*)calloc((size_t)N, sizeof(float));
    if (!col_sums) {
        perror("calloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float val = fabsf(A[i * N + j]);
            row_sum += val;
            col_sums[j] += val;
        }
        if (row_sum > *norm_inf) {
            *norm_inf = row_sum;
        }
    }

    for (int j = 0; j < N; ++j) {
        if (col_sums[j] > *norm1) {
            *norm1 = col_sums[j];
        }
    }

    free(col_sums);
}

void generate_test_matrix(float* A, int N) {
    // Матрица с диагональным преобладанием для быстрой сходимости ряда
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (i == j) ? (float)(N + 10) : 0.5f;
        }
    }
}

int read_input(int* N, int* M, float** A) {
    printf("Введите N (размер матрицы) и M (число членов ряда): ");
    if (scanf("%d %d", N, M) != 2 || *N <= 0 || *M <= 0) {
        fprintf(stderr, "Ошибка ввода N и M.\n");
        return -1;
    }

    *A = alloc_matrix(*N);
    printf("Введите элементы матрицы %dx%d построчно:\n", *N, *N);
    for (int i = 0; i < (*N) * (*N); ++i) {
        if (scanf("%f", &(*A)[i]) != 1) {
            fprintf(stderr, "Ошибка чтения элемента матрицы.\n");
            free(*A);
            return -1;
        }
    }
    return 0;
}

void print_matrix(const char* name, const float* A, int N) {
    printf("\n%s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%10.5f ", A[i * N + j]);
        }
        printf("\n");
    }
}

// Вычисление невязки ||A * A_inv - I||_inf
float compute_residual(const float* A, const float* A_inv, int N) {
    float* prod = alloc_matrix(N);
    zero_matrix(prod, N);

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                prod[i * N + j] += a_ik * A_inv[k * N + j];
            }
        }
    }

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        prod[i * N + i] -= 1.0f;
        float row_err = 0.0f;
        for (int j = 0; j < N; ++j) {
            row_err += fabsf(prod[i * N + j]);
        }
        if (!isfinite(row_err)) {
            max_err = INFINITY;
            break;
        }
        if (row_err > max_err) {
            max_err = row_err;
        }
    }

    free(prod);
    return max_err;
}
