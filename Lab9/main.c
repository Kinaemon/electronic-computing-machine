#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <sched.h>
#include <sys/mman.h>

#define L1D_SIZE         (48 * 1024)        // Точный размер L1d для i7-13700H (P-core)
#define MAX_FRAGMENTS    32
#define ITERATIONS       20000000ULL

typedef uint32_t elem_t;

// Инициализация цепочки обхода с заданным шагом между фрагментами (offset_bytes)
static void init_array(elem_t *arr, size_t n_fragments, size_t offset_bytes) {
    size_t offset = offset_bytes / sizeof(elem_t);
    size_t total_elements = L1D_SIZE / sizeof(elem_t);
    size_t frag_elements = total_elements / n_fragments;

    // Шаг 16 элементов = 64 байта (ровно 1 кэш-линия, всего 64 набора в L1d)
    size_t stride = 16;
    size_t num_lines = frag_elements / stride;
    if (num_lines == 0) num_lines = 1;

    for (size_t line = 0; line < num_lines; ++line) {
        // Небольшое перемешивание порядка наборов против аппаратного стримингового префетчера
        size_t cur_set = (line * 17) % num_lines;
        size_t next_set = ((line + 1) * 17) % num_lines;

        size_t offset_in_frag = cur_set * stride;
        size_t next_offset_in_frag = next_set * stride;

        for (size_t j = 0; j < n_fragments - 1; ++j) {
            arr[j * offset + offset_in_frag] = (elem_t)((j + 1) * offset + offset_in_frag);
        }
        arr[(n_fragments - 1) * offset + offset_in_frag] = (elem_t)next_offset_in_frag;
    }
}

static double measure(elem_t *arr, size_t n_fragments, size_t offset_bytes) {
    init_array(arr, n_fragments, offset_bytes);

    // Разогрев кэша и вывод ядра на рабочую частоту
    volatile elem_t k = 0;
    for (size_t i = 0; i < 500000; ++i) {
        k = arr[k];
    }

    unsigned int aux = 0;
    elem_t idx = 0;
    uint64_t best_cycles = UINT64_MAX;

    for (int run = 0; run < 5; ++run) {
        _mm_mfence();
        uint64_t start = __rdtscp(&aux);

        for (size_t i = 0; i < ITERATIONS / 8; ++i) {
            idx = arr[idx];
            idx = arr[idx];
            idx = arr[idx];
            idx = arr[idx];
            idx = arr[idx];
            idx = arr[idx];
            idx = arr[idx];
            idx = arr[idx];
        }

        uint64_t end = __rdtscp(&aux);
        _mm_mfence();

        if (end - start < best_cycles) {
            best_cycles = end - start;
        }
    }

    k = idx;
    (void)k;
    return (double)best_cycles / (double)ITERATIONS;
}

int main(void) {
    // Фиксируем поток на ядре 0 (P-core с L1d = 48 KB, 12-way)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);

    size_t max_offset = 2 * 1024 * 1024; // 2 МБ
    size_t total_bytes = (size_t)MAX_FRAGMENTS * max_offset;

    elem_t *arr = NULL;
    if (posix_memalign((void **)&arr, 2 * 1024 * 1024, total_bytes) != 0 || !arr) {
        perror("posix_memalign");
        return 1;
    }

    madvise(arr, total_bytes, MADV_HUGEPAGE);

    // Заполняем память для предотвращения page faults
    for (size_t i = 0; i < total_bytes / sizeof(elem_t); i += 1024) {
        arr[i] = 0;
    }

    printf("Fragments,Cycles_L1_to_L2,Cycles_L1_L2_to_L3\n");

    for (size_t n = 1; n <= MAX_FRAGMENTS; ++n) {
        // 1. Шаг 64 КБ: бьёт в те же наборы L1d (кратно 4 КБ), но в разные наборы L2 (не кратно 128 КБ)
        double c_l1 = measure(arr, n, 64 * 1024);
        // 2. Шаг 2 МБ: бьёт в один и тот же набор и в L1d, и в L2 (кратно 128 КБ внутри HugePage)
        double c_l2 = measure(arr, n, 2 * 1024 * 1024);

        printf("%zu,%.2f,%.2f\n", n, c_l1, c_l2);
        fflush(stdout);
    }

    free(arr);
    return 0;
}