#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifdef __linux__
#include <cpuid.h>
#include <sched.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// Глобальная переменная с квалификатором volatile не дает компилятору
// удалить цикл обхода при включенной оптимизации -O1
volatile int sink = 0;

static int pin_to_performance_core(void) {
    unsigned int eax, ebx, ecx, edx;
    // CPUID.07H.0H:EDX[15] указывает на гибридную архитектуру Intel.
    if (!__get_cpuid_count(0x07, 0, &eax, &ebx, &ecx, &edx)
        || !(edx & (1u << 15))) {
        return 0;
    }

    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == -1) {
        perror("sched_getaffinity");
        return -1;
    }

    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpu, &affinity);
        if (sched_setaffinity(0, sizeof(affinity), &affinity) == -1) {
            continue;
        }

        // Тип ядра определяем на самом проверяемом CPU. 40H — Intel Core (P-ядро).
        if (__get_cpuid_count(0x1a, 0, &eax, &ebx, &ecx, &edx)
            && (eax >> 24) == 0x40) {
            fprintf(stderr, "Benchmark CPU: %d (P-core)\n", cpu);
            return 0;
        }
    }

    fprintf(stderr, "No accessible Intel P-core for the benchmark\n");
    return -1;
}

// 1. Прямой обход: 0 -> 1 -> 2 -> ... -> N-1 -> 0
void fill_forward(int *arr, size_t n) {
    for (size_t i = 0; i < n - 1; i++) {
        arr[i] = (int)(i + 1);
    }
    arr[n - 1] = 0;
}

// 2. Обратный обход: 0 -> N-1 -> N-2 -> ... -> 1 -> 0
void fill_backward(int *arr, size_t n) {
    arr[0] = (int)(n - 1);
    for (size_t i = 1; i < n; i++) {
        arr[i] = (int)(i - 1);
    }
}

// 3. Случайный обход (Алгоритм Саттоло для создания единого цикла длины N)
void fill_random(int *arr, size_t n, int *perm) {
    for (size_t i = 0; i < n; i++) {
        perm[i] = (int)i;
    }
    // Перемешиваем индексы так, чтобы гарантированно получился один цикл
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = ((size_t)rand() * RAND_MAX + (size_t)rand()) % i;
        int temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
    for (size_t i = 0; i < n - 1; i++) {
        arr[perm[i]] = perm[i + 1];
    }
    arr[perm[n - 1]] = perm[0];
}

// Измерение времени обхода в тактах процессора на один элемент
double measure(int *arr, size_t n, size_t k_passes, int repeats) {
    // Примечание 2: Однократный обход для «прогрева» кэш-памяти
    int k = 0;
    for (size_t i = 0; i < n; i++) {
        k = arr[k];
    }
    sink = k;

    uint64_t min_ticks = UINT64_MAX;
    size_t total_steps = n * k_passes;

    // Примечание 3: Несколько замеров с выбором минимального времени
    for (int r = 0; r < repeats; r++) {
        k = 0;
        uint64_t start = __rdtsc();
        for (size_t i = 0; i < total_steps; i++) {
            k = arr[k];
        }
        uint64_t end = __rdtsc();
        sink = k;

        uint64_t ticks = end - start;
        if (ticks < min_ticks) {
            min_ticks = ticks;
        }
    }

    return (double)min_ticks / (double)total_steps;
}

int main(void) {
    if (pin_to_performance_core() == -1) {
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));

    // Границы: от 1 КБ (256 элементов int) до 64 МБ (16 777 216 элементов int)
    size_t n_min = 256;
    size_t n_max = 16 * 1024 * 1024;
    int repeats = 5; // Число повторов для фильтрации выбросов

    int *arr = (int *)malloc(n_max * sizeof(int));
    int *perm = (int *)malloc(n_max * sizeof(int));
    if (!arr || !perm) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    printf("Size_Bytes,Size_KB,Forward_Cycles,Backward_Cycles,Random_Cycles\n");

    // Переменный шаг: увеличиваем размер примерно на 20% на каждой итерации
    for (size_t n = n_min; n <= n_max; ) {
        // Адаптируем число проходов K, чтобы мелкие массивы крутились больше раз,
        // а крупные не зависали надолго
        size_t total_ops = 2 * 1024 * 1024;
        size_t k_passes = total_ops / n;
        if (k_passes < 2) k_passes = 2;

        fill_forward(arr, n);
        double t_fwd = measure(arr, n, k_passes, repeats);

        fill_backward(arr, n);
        double t_bwd = measure(arr, n, k_passes, repeats);

        fill_random(arr, n, perm);
        double t_rnd = measure(arr, n, k_passes, repeats);

        size_t bytes = n * sizeof(int);
        double kb = (double)bytes / 1024.0;
        printf("%zu,%.2f,%.2f,%.2f,%.2f\n", bytes, kb, t_fwd, t_bwd, t_rnd);
        fflush(stdout);

        size_t next_n = (size_t)(n * 1.2);
        if (next_n <= n) next_n = n + 1;
        n = next_n;
    }

    free(arr);
    free(perm);
    return 0;
}
