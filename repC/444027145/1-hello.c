/*
A parallel hello world program that uses OpenMP

nproc - print the number of processing units available
lscpu - display information about the CPU architecture

printenv
Print the values of the specified environment VARIABLE(s)
printenv OMP_NUM_THREADS

export OMP_NUM_THREADS=4 (win: set OMP_NUM_THREADS=4)
unset OMP_NUM_THREADS    (win: set OMP_NUM_THREADS=)

Compile:
    gcc -g -Wall -fopenmp -o 1-hello 1-hello.c
    or
    gcc -fopenmp -o <filename> <file>.c

for mac use gcc-9

https://gcc.gnu.org/wiki/openmp
gcc OpenMP support
*/
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// for openmp

// thread function
// void hello(void);
//----------------
void hello(void)
{
    // returns the thread number within the current team
    int my_rank = omp_get_thread_num();

    // returns the number of threads in the current team
    int thread_count = omp_get_num_threads();

    printf("--Hello--");
    printf("from thread %d of %d\n", my_rank, thread_count);
}
//--------------------------------
int main(int argc, char *argv[])
{
    int thread_count = 4;

    // get number of threads from command line
    if (argc > 1)
        thread_count = atoi(argv[1]);

        // sets the number of threads for parallel regions
        // omp_set_num_threads(7);

        // returns the number of processors available to the device
        int p = omp_get_num_procs();
        printf("p = %d\n", p);

        // ?
        // int t = omp_get_num_threads();
        // printf("t = %d\n", t);

#pragma omp parallel num_threads(thread_count)
    {
        hello();
    }

    printf("done\n");
    return 0;
}
