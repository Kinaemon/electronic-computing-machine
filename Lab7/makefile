CC = gcc
CFLAGS_BASE = -std=c11 -O3 -Wall -Wextra
CFLAGS_SCALAR = $(CFLAGS_BASE) -fno-tree-vectorize
CFLAGS_AUTO = $(CFLAGS_BASE) -mavx2 -mfma -ftree-vectorize
CFLAGS_SIMD = $(CFLAGS_BASE) -mavx2 -mfma
CFLAGS_BLAS = $(CFLAGS_BASE)
LDFLAGS = -lm

TARGETS = inv_scalar inv_auto inv_simd inv_blas

.PHONY: all bench check clean

all: $(TARGETS)

inv_scalar: main_scalar.c common.c common.h makefile
	$(CC) $(CFLAGS_SCALAR) main_scalar.c common.c -o inv_scalar $(LDFLAGS)

inv_auto: main_scalar.c common.c common.h makefile
	$(CC) $(CFLAGS_AUTO) -DVARIANT_NAME='"Auto"' main_scalar.c common.c -o inv_auto $(LDFLAGS)

inv_simd: main_simd.c common.c common.h makefile
	$(CC) $(CFLAGS_SIMD) main_simd.c common.c -o inv_simd $(LDFLAGS)

inv_blas: main_blas.c common.c common.h makefile
	$(CC) $(CFLAGS_BLAS) main_blas.c common.c -o inv_blas -lopenblas $(LDFLAGS)

bench: all
	@echo "=== Сравнение при N=2048, M=10 (1 поток OpenBLAS) ==="
	OPENBLAS_NUM_THREADS=1 ./inv_scalar 2048 10
	OPENBLAS_NUM_THREADS=1 ./inv_auto 2048 10
	OPENBLAS_NUM_THREADS=1 ./inv_simd 2048 10
	OPENBLAS_NUM_THREADS=1 ./inv_blas 2048 10

check: all
	@echo "=== Проверка невязки (не более 1e-4) ==="
	@set -e; \
	for target in $(TARGETS); do \
		small=$$(printf '%s\n' '3 120' '4 1 0' '0 3 1' '0 0 2' | OPENBLAS_NUM_THREADS=1 ./$$target); \
		tail=$$(OPENBLAS_NUM_THREADS=1 ./$$target 27 40); \
		printf '%s\n' "$$small" "$$tail" | awk '/Residual/ { print; if ($$NF ~ /^[0-9]/ && $$NF <= 1e-4) ++passed } END { exit (passed != 2) }'; \
	done

clean:
	rm -f $(TARGETS)
