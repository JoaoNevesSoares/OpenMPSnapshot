//transpose.c
//serial code for transposing a matrix in-place
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define SEED 0
#define n 5000
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

void free_matrix(double **a) {
	int i;
	for (i=0;i<n;i++) {
		free(a[i]);
	}
	free(a);
}

int main(int argc, char *argv[]) {
	//SETUP CODE
        int i,j,ii,jj;
	double **A;
	double t1,t2;
	int numthreads,tid;

        #pragma omp parallel
        {
                numthreads = omp_get_num_threads();
                tid = omp_get_thread_num();
                if(tid==0)
                        printf("Running transpose with %d threads\n",numthreads);
        }

	A = create_matrix();

        /* Creating matrix B which will contained the transposed version of A */
	double** B = (double**) malloc(sizeof(double*)*n);
	for (i=0;i<n;i++) {
		B[i] = (double*) malloc(sizeof(double)*n);
	}
       
        int bsize; 
        for(bsize = 1 ; bsize < 5000 ; bsize = bsize * 2) {
            printf("Working with bsize = %d\t", bsize);
	    t1 = get_clock();

	    //BEGIN MAIN ROUTINE
            for (ii = 0; ii<n; ii+=bsize) {
              for (jj = 0; jj<n; jj+=bsize) {
	        for(i=ii;i < MIN(ii+bsize, n); i++) {
		  for(j=jj ;j<MIN(jj+bsize,n);j++) {
			B[i][j] = A[j][i];
	          }
                }
	      }
            }
	//END MAIN ROUTINE

	    t2 = get_clock();
	    printf("Time: %lf\n",(t2-t1));
        }

	if(SAVE) {
                // Output Result
                char outfile[100];
                sprintf(outfile,"transpose_out_%d.txt",numthreads);
                printf("Outputting solution to %s\n",outfile);
                FILE *fp = fopen(outfile,"w");
                for(i=0; i<n; i++) {
			for(j=0; j<n; j++) {
                        	fprintf(fp,"%lf\n",B[i][j]);
			}
		}
                fclose(fp);
        }

	//CLEANUP CODE
	free_matrix(A);
	return 0;

}
