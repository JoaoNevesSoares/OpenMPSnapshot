/*

   This program will numerically compute the integral of

   4/(1+x*x) 

   from 0 to 1.  The value of this integral is pi -- which 
   is great since it gives us an easy way to check the answer.

   The is the original sequential program.  It uses the timer
   from the OpenMP runtime library

*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
static long num_steps = 100000000;
double step;
int main (int argc, char *argv[])
{

	int nthreads;
	//	unsigned int thread_qty = atoi(getenv("OMP_NUM_THREADS"));
	unsigned int thread_qty = atoi(argv[1]);
	//	printf("requested threads: %d\n",thread_qty);
	omp_set_num_threads(thread_qty);
	double start_time, run_time;
	double pi =0.0;

	step = 1.0/(double) num_steps;

	start_time = omp_get_wtime();

#pragma omp parallel
	{
		int i;
		double sum;
		double x;
		int id = omp_get_thread_num();
		int num_threads =  omp_get_num_threads();   	 
		if(id == 0)
			nthreads = num_threads;
		for (i=id, sum = 0.0; i< num_steps; i+=num_threads){
			x = (i+0.5)*step;
			sum = sum + 4.0/(1.0+x*x);
		}
		sum = sum * step;
#pragma omp atomic
		pi += sum;
	}

	int i;
	//printf("given threads: %d\n",nthreads);
	for(i = 0; i < nthreads;i++)
		run_time = omp_get_wtime() - start_time;
	//printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,run_time);
	printf("%lf\n",run_time);
}	  
