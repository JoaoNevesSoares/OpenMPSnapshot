#include <omp.h>
#include <stdio.h>
int main(){
int x = 2;
#pragma omp task shared(x) mergeable
{
x++;
}
#pragma omp taskwait
printf("%d\n",x);
return 0;
}