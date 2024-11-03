/*
Use odd-even transposition sort to sort a list of ints.
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Keys in the random list in the range 0 <= key < RMAX */
const int RMAX = 100;

void odd_even_sort(int a[], int n, int);

//--------------------------------
int main(int argc, char* argv[]) {
    int thread_count = 4;
    int n = 20;

    if (argc > 2){
        thread_count = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    int* a = malloc(n * sizeof(int));

    // generate_list
    srand(0);
    for (int i = 0; i < n; i++)
        a[i] = rand() % RMAX;
    
    // print_list
    if (n <= 20) {
        for (int i = 0; i < n; i++)
            printf("%2d ", a[i]);
        printf("before\n");
    }

    // time
    double t0 = omp_get_wtime();
    odd_even_sort(a, n, thread_count);
    
    double t1 = omp_get_wtime();
    printf("time = %.3f sec\n", t1 - t0);

    // print_list
    if (n <= 20) {
        for (int i = 0; i < n; i++)
            printf("%2d ", a[i]);
        printf("after\n");
    }

    free(a);
    return 0;
}

/*-----------------------------------------------------------------
 * Function:     Odd_even_sort
 * Purpose:      Sort list using odd-even transposition sort
 * In args:      n
 * In/out args:  a
 */
void odd_even_sort(int *a, int n, int thread_count) {
    int phase, i, temp;

    // #pragma omp parallel num_threads(thread_count) \
            default(none) shared(a, n) private(i, temp, phase)
    for (phase = 0; phase < n; phase++) 
        if (phase % 2 == 0) {   /* even phase */
            // #pragma omp for        
            #pragma omp parallel for num_threads(thread_count) \
                    default(none) shared(a, n) private(i, temp)
            for (i = 1; i < n; i += 2) 
                if (a[i-1] > a[i]) {
                    temp = a[i];
                    a[i] = a[i-1];
                    a[i-1] = temp;
                }
        } else {                /* odd phase */
            // #pragma omp for        
            #pragma omp parallel for num_threads(thread_count) \
                    default(none) shared(a, n) private(i, temp)
            for (i = 1; i < n-1; i += 2)
                if (a[i] > a[i+1]) {
                    temp = a[i];
                    a[i] = a[i+1];
                    a[i+1] = temp;
                }
        }
}
