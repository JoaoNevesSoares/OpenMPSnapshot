#include <omp.h>
#include<stdio.h>
int main ()
{
    int nthreads, tid;

    #pragma omp parallel private(tid)
    {
    tid = omp_get_thread_num();
    printf("Your Name\n");
    if (tid == 0) 
    { 
        nthreads =omp_get_num_threads(); 
        printf("Registration Number\n"); 
        } 
    }   
}
