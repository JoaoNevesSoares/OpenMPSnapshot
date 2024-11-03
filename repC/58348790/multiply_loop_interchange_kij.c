//multiply.c
//serial code for multiplying two nxn matrices
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#define SEED 0
#define n 1000
#define SAVE 1

struct timeval tv; 
double get_clock() {
   struct timeval tv; int ok;
   ok = gettimeofday(&tv, (void *) 0);
   if (ok<0) { printf("gettimeofday error");  }
   return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6); 
}

double **create_matrix() {
	int i,j;
	double **a;
	a = (double**) malloc(sizeof(double*)*n);
	for (i=0;i<n;i++) {
		a[i] = (double*) malloc(sizeof(double)*n);
	}

	srand(SEED);
	for (i=0;i<n;i++) {
		for (j=0;j<n;j++) {
			a[i][j] = rand()%10;
		}
	}

	return a;
}

void free_matrix(double** a) {
	int i;
	for (i=0;i<n;i++) {
		free(a[i]);
	}
	free(a);
}

int main(int argc, char *argv[]) {
	//SETUP CODE
	int i,j,k;
	double **A,**B,**C;
	double t1,t2;
	int numthreads,tid;
	
	#pragma omp parallel
        {
                numthreads = omp_get_num_threads();
                tid = omp_get_thread_num();
                if(tid==0)
                        printf("Running multiply with %d threads\n",numthreads);
        }

	A = create_matrix();
	B = create_matrix();
	C = (double**) malloc(sizeof(double*)*n);
	for (i=0;i<n;i++) {
		C[i] = (double*) malloc(sizeof(double)*n);
	}
	//END SETUP CODE
	for(i=0;i<n;i++) {
		for(j=0;j<n;j++) {
                    C[i][j] = 0;
		}
	}

	t1 = get_clock();

	//BEGIN MAIN ROUTINE
	for(k=0;k<n;k++) {
		for(i=0;i<n;i++) {
			for(j=0;j<n;j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	//END MAIN ROUTINE

	t2 = get_clock();
	printf("Time: %lf\n",(t2-t1));

        if(SAVE) {
                // Output Result
                char outfile[100];
                sprintf(outfile,"kij_%d.txt",numthreads);
                printf("Outputting solution to %s\n",outfile);
                FILE *fp = fopen(outfile,"w");
                for(i=0; i<n; i++) {
			for(j=0; j<n; j++) {
                       		fprintf(fp,"%lf\n",C[i][j]);
			}
		}
                fclose(fp);
        }

	//CLEANUP CODE
	free_matrix(A);
	free_matrix(B);
	free_matrix(C);
	return 0;

}