#define _GNU_SOURCE // To remove implicit declaration of function ‘sched_getcpu’
#include <stdio.h>
#include <sched.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Change the size here and run myscript.sh*/
#define SIZE 10000000

int arr[SIZE];
int arrcopy[SIZE];
int tmp[SIZE];
int *orig;

void printarr(int *arr, int n);
void mergesort_omp_parallel(int *arr, int *tmp, int size, int threads);
void mergesort_serialize(int *arr, int *tmp, int size);
void merge(int *arr, int size, int *tmp);

void printarr(int *arr, int n){
	for(int i = 0;i < n; i++)
		printf("%d ", arr[i]);
	printf("\n");
}

void merge(int *arr, int size, int *tmp){
	int i = 0;
	int j = size/2;
	int ti = 0;

	while(i < size/2 && j < size){
		if(arr[i] < arr[j]){
			tmp[ti] = arr[i];
			i++;
		}
		else{
			tmp[ti] = arr[j];
			j++;
		}
		ti++;
	}
	while(i < size/2){
		tmp[ti] = arr[i];
		ti++; i++;
	}
	while(j < size){
		tmp[ti] = arr[j];
		ti++; j++;
	}
	memcpy(arr, tmp, size*sizeof(int));
}

void mergesort_omp_parallel(int *arr, int *tmp, int size, int threads){
	if(size <= 1)
		return;

	if(threads == 1){
		int thread_num = omp_get_thread_num();
		int cpu_num = sched_getcpu();
		int l = arr - orig;
		int r = l + size;
		printf("Range [%8d %8d) is sorted by thread %2d is running on CPU %2d\n", l, r, thread_num,cpu_num);
		mergesort_serialize(arr, tmp, size);
	}

	else
	{
#pragma omp parallel sections
		{
#pragma omp section
			{
				mergesort_omp_parallel(arr, tmp, size/2, threads/2);
			}
#pragma omp section
			{
				mergesort_omp_parallel(arr + (size/2), tmp + size / 2, size - size/2, threads - threads/2);
			}
		}
		merge(arr, size, tmp);
	}
}

void mergesort_serialize(int *arr, int *tmp, int size){
	if(size <= 1)
		return;
	mergesort_serialize(arr, tmp, size/2);
	mergesort_serialize(arr + (size/2), tmp + size / 2, size - size/2);
	merge(arr, size, tmp);
}


int main(int argc, char *argv[])
{
	if(argc < 2){
		printf("Usage ./a.out <number_of_threads>\n");
		exit(1);
	}
	orig = arr;

	int nthreads;
	unsigned int thread_qty = atoi(argv[1]);
	omp_set_num_threads(thread_qty);
	int i;
	double start_time, run_time;

	/* Initialize */
	srand(time(NULL));
	for(int i=0; i < SIZE; i++)
	{
		arr[i] = rand() % SIZE;
		arrcopy[i] = arr[i];
	}
	int threads;

#pragma omp parallel
	{
#pragma omp single
		{
			threads = omp_get_num_threads();
		}
	}
	start_time = omp_get_wtime();
	mergesort_omp_parallel(arr, tmp, SIZE, threads);
	run_time = omp_get_wtime() - start_time;
	printf("%f\n", run_time);
	//printf(" Time to sort(in parallel) Array of size %d is %f seconds \n", SIZE, run_time);

	start_time = omp_get_wtime();
	mergesort_serialize(arrcopy, tmp, SIZE);
	run_time = omp_get_wtime() - start_time;
	printf(" Time to sort(in serial) Array of size %d is %f seconds \n", SIZE, run_time);
	/* TERMINATE PROGRAM */
	//
	for (i = 1; i < SIZE; i++)
	{
		if (!(arr[i - 1] <= arr[i]))
		{
			printf ("Implementation error //el: a[%d]=%d > a[%d]=%d\n", i - 1,
					arr[i - 1], i, arr[i]);
			return 1;
		}
	}

	for (i = 1; i < SIZE; i++)
	{
		if (!(arrcopy[i - 1] <= arrcopy[i]))
		{
			printf ("Implementation error serial: a[%d]=%d > a[%d]=%d\n", i - 1,
					arrcopy[i - 1], i, arrcopy[i]);
			return 1;
		}
	}

	return 0;
}

