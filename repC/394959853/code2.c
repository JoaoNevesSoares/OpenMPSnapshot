#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main()
{
    int x = 0,y=0;
    #pragma omp parallel num_threads(60)
    {
        #pragma omp single
        x++;
        #pragma omp critical 
        y++;
    }
    printf("Single: %d\n", x);
    printf("Parallel: %d\n",y);
}
