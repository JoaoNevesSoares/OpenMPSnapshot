#include <stdio.h>
#include <omp.h>
int
main(void)
{
int a,i;
#pragma omp master
{
a=0;
}
omp_set_num_threads(4);
#pragma omp parallel shared(a) private(i)
{
#pragma omp for reduction(+:a)
for(i=0;i<10;i++)
a+=i;
#pragma omp single nowait
printf("Valor de a:%d en el thread %d\n",a,omp_get_thread_num());
}
printf("La suma es %d\n",a);
return 0;
}