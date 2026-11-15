CC = gcc
CFLAGS_BASE = -std=c11 -O3 -Wall -Wextra
CFLAGS_SCALAR = $(CFLAGS_BASE) -fno-tree-vectorize
CFLAGS_SIMD = $(CFLAGS_BASE) -mavx2 -mfma
CFLAGS_BLAS = $(CFLAGS_BASE)
LDFLAGS = -lm

TARGETS = inv_scalar inv_simd inv_blas

all: $(TARGETS)

inv_scalar: main_scalar.c common.c common.h
	$(CC) $(CFLAGS_SCALAR) main_scalar.c common.c -o inv_scalar $(LDFLAGS)

inv_simd: main_simd.c common.c common.h
	$(CC) $(CFLAGS_SIMD) main_simd.c common.c -o inv_simd $(LDFLAGS)

inv_blas: main_blas.c common.c common.h
	$(CC) $(CFLAGS_BLAS) main_blas.c common.c -o inv_blas -lopenblas $(LDFLAGS)
	
bench: all
	@echo "=== Сравнение при N=2048, M=10 (1 поток OpenBLAS) ==="
	OPENBLAS_NUM_THREADS=1 ./inv_scalar 2048 10
	OPENBLAS_NUM_THREADS=1 ./inv_simd 2048 10
	OPENBLAS_NUM_THREADS=1 ./inv_blas 2048 10

clean:
	rm -f $(TARGETS)