// OpenMP code to get first n prime numbers
// Note: Refer README file for more information.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 1000000
#define N_THREADS 8

// Function to check if a number is prime
int isPrime(int n) {
    int i;
    for (i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) {
            return 0;
        }
    }
    return 1;
}
// End of isPrime()

// Main function
int main() {
    // Declare variables
    int n = N;
    int i, count = 0; 
    struct timeval start, stop;
    double cpu_time_used;
    
    printf("OpenMP code to get first %d prime numbers\n", n);
    printf("Threads: %d\n\n", omp_get_max_threads());

    // OpenMP parallel code segment
    mingw_gettimeofday(&start, 0);
    #pragma omp parallel for num_threads(N_THREADS) reduction(+:count)
    for (i = 2; i <= n; i++) {
        if (isPrime(i)) {
            count++;
        }
    }
    mingw_gettimeofday(&stop, 0);

    // Calculating time taken by the OpenMP parallel code segment
    cpu_time_used = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("Time taken: %f", cpu_time_used);

    return 0;
}
// End of main()
// End of omp.c