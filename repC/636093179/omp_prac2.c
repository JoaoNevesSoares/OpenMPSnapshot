#include<stdio.h>
#include<omp.h>

int main(){
	int partial_Sum, total_Sum;

	#pragma omp parallel private(partial_Sum) shared(total_Sum)
	{
		partial_Sum = 0;
		total_Sum = 0;

		#pragma omp for
		for(int i = 1; i <=1000; i++){
			partial_Sum += i;
		}

		#pragma omp critical
		{
			total_Sum += partial_Sum;
		}
	}
	printf("Total Sum: %d\n", total_Sum);
	return 0;
}