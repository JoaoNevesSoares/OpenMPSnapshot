#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#include "Solve_Systems.cuh"

// Per assicurarmi di non eccedere il limite dei 1024 blocchi della "Tesla T4"
#define TILE 16



/************************************************************************/
/*******| Funzione per effettuare la Decomposizione LU sulla GPU |*******/
/************************************************************************/

void LUDecompose(double* gpu_a, int n, int numblock) {

    // Itera attraverso le colonne della matrice
    for (int i = 0; i < n; ++i) {

        scala_Indice<<<1, 1>>>(gpu_a, n, i);
        eliminazione_gaussiana<<<numblock, TILE, n * sizeof(double)>>>(gpu_a, n, i, TILE);

    }

}


// Kernel CUDA per scalare la riga corrente
__global__ void scala_Indice(double *matrix, int n, int index) {

    int start = (index * n + index);
    int end = (index * n + n);

    // Normalizzazione Doolittle (LU) --> Divide ogni elemento della riga per il pivot
    for (int i = start + 1; i < end; ++i) {
        matrix[i] = (matrix[i] / matrix[start]);
    }

}


// Kernel CUDA per l'eliminazione gaussiana
__global__ void eliminazione_gaussiana(double *A, int n, int index, int bsize) {

    extern __shared__ double pivot[];

    int idThread = threadIdx.x;
    int idBlock = blockIdx.x;
    int blockSize = bsize;

    // Copia il pivot nella memoria condivisa
    if (idThread == 0) {

        for (int i = index; i < n; i++) pivot[i] = A[(index * n) + i];

    }


    // Aspetto che tutti i thread del blocco siano terminati
    __syncthreads();


    int pivotRow = (index * n);
    int currentRow = (((blockSize * idBlock) + idThread) * n);
    int start = currentRow + index;
    int end = currentRow + n;

    // Esegue l'eliminazione gaussiana sui blocchi paralleli
    if (currentRow > pivotRow) {

        for (int i = start + 1; i < end; ++i) {

            A[i] = A[i] - (A[start] * pivot[i - currentRow]);

        }

    }

}




/*************************************************************************/
/*******|  Funzione per risolvere il sistema lineare (sulla CPU)  |*******/
/*************************************************************************/

int LUSolve(int n, double** L, double** U, double* b) {

    // Imposta il numero di core della CPU utilizzati per la parallelizzazione
    int NUM_CORES = omp_get_num_procs();
    omp_set_num_threads(NUM_CORES);


    // Forward substitution (Ly = b)
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i][j] * b[j];
        }
        b[i] = sum / L[i][i];
    }


    // Backward substitution (Ux = y)
    #pragma omp parallel for
    for (int i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= U[i][j] * b[j];
        }
        b[i] = sum / U[i][i];
    }


    return NUM_CORES;

}




/************************************************************************/
/*******|  Funzione di utilità per gestire le matrici dinamiche  |*******/
/************************************************************************/

void generaMatrice(double* a, int n) {

    // Inizializza la matrice con numeri casuali compresi tra -100 e 100
    for (int i = 0; i <= (n * n); ++i) {
        a[i] = ((rand() % 201) - 100);
    }

    int diagCount = 0;
    double sum = 0;

    // Imposta i valori sulla diagonale in modo che la matrice sia diagonale dominante
    for (int i = 0; i < n; ++i) {
        for (int j = i * n; j < i * n + n; ++j) {
            sum += abs(a[j]);
        }
        sum -= abs(a[i * n + diagCount]);
        a[i * n + diagCount] = sum + ((rand() % 5) + 1);
        ++diagCount;
        sum = 0;
    }

}


void initialize_matrices(double** a, double** l, double** u, int size) {

    for (int i = 0; i < size; ++i) {
        a[i] = (double*)malloc(size * sizeof(double));
        l[i] = (double*)malloc(size * sizeof(double));
        u[i] = (double*)malloc(size * sizeof(double));
    }

}


void deallocate_matrices(double** a, double** l, double** u, int size) {

    for (int i = 0; i < size; ++i) {
        free(a[i]);
        free(l[i]);
        free(u[i]);
    }

    free(a);
    free(l);
    free(u);

}