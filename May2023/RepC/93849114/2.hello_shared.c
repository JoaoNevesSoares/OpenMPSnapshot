#include <stdio.h>
#include <omp.h>
int main ()
{
int id;
#pragma omp parallel shared(id)
{
id =omp_get_thread_num();
printf("(%d) Hello ",id);
printf("(%d) world!\n",id);
}
return 0;
}