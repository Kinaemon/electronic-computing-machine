#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>

// Выделение выровненной по 32 байта памяти (под AVX2)
float* alloc_matrix(int N);
void zero_matrix(float* A, int N);
void set_identity(float* I, int N);

// Вычисление норм ||A||_1 и ||A||_inf
void compute_norms(const float* A, int N, float* norm1, float* norm_inf);

// Ввод/вывод и тестирование
double get_time_sec(void);
void generate_test_matrix(float* A, int N);
int read_input(int* N, int* M, float** A);
void print_matrix(const char* name, const float* A, int N);
float compute_residual(const float* A, const float* A_inv, int N);

#endif // COMMON_H