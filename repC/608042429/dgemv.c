#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <inttypes.h>
#include "omp.h"

int m, n;

double wtime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

/* matrix_vector_product_omp: Compute matrix-vector product c[m] = a[m][n] * b[n] */
void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
	#pragma omp parallel //num_threads(8)
	{
		int nthreads = omp_get_num_threads();
		int threadid = omp_get_thread_num();
		//printf("Thread %d of %d\n", threadid, nthreads);
		int items_per_thread = m / nthreads;
		int lb = threadid * items_per_thread;
		int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
		for (int i = lb; i <= ub; i++)
		{
			c[i] = 0.0;
			for (int j = 0; j < n; j++)
				c[i] = c[i] + a[i * n + j] * b[j];
		}
	}
}

void run_parallel()
{
	double *a, *b, *c;

	a = malloc(sizeof(*a) * n * m);
	b = malloc(sizeof(*b) * n);
	c = malloc(sizeof(*c) * m);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		a[i * n + j] = i + j;
	}

	for (int j = 0; j < n; j++)
		b[j] = j;

	double t = wtime();
	matrix_vector_product_omp(a, b, c, m, n);
	t = wtime() - t;

	printf("Elapsed time (parallel): %.7f sec.\n", t);
	free(a);
	free(b);
	free(c);
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		c[i] = 0.0;
		for (int j = 0; j < n; j++)
			c[i] += a[i * n + j] * b[j];
	}
}

void run_serial()
{
	double *a, *b, *c;

	a = malloc(sizeof(*a) * m * n);
	b = malloc(sizeof(*b) * n);
	c = malloc(sizeof(*c) * m);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			a[i * n + j] = i + j;
	}
	for (int j = 0; j < n; j++)
		b[j] = j;

	double t = wtime();
	matrix_vector_product(a, b, c, m, n);
	t = wtime() - t;

	printf("Elapsed time (serial): %.7f sec.\n", t);
	free(a);
	free(b);
	free(c);
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		printf("Not enough arguments\n");
		return 0;
	}

	m = atoi(argv[1]);
	n = atoi(argv[2]);

	printf("MATRIX = %d\n", m * n);
	printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
	printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);

	run_serial();
	run_parallel();

	return 0;
}