// first_omp_exemple.c
// compile with: /openmp

/* #############################################################################
## DESCRIPTION: Simple exemple with OpenMp.
## NAME: first_omp_exemple.c
## AUTHOR: Lucca Pessoa da Silva Matos
## DATE: 10.04.2020
## VERSION: 1.0
## EXEMPLE:
##     PS C:\> gcc -fopenmp -o first_omp_exemple first_omp_exemple.c
##############################################################################*/

// =============================================================================
// LIBRARYS
// =============================================================================

#include <omp.h>
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>

#define NUM_THREADS 12

// =============================================================================
// CALL FUNCTIONS
// =============================================================================

void cabecalho();
void set_portuguese();

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char const *argv[]){

  set_portuguese();
  cabecalho();

  omp_set_num_threads(NUM_THREADS);

  printf("\n1.1 - Estamos fora do contexto paralelo...\n\n");

  // Fork
  #pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    printf("Eu sou a Thread %d de um total de %d\n", thread_id, num_threads);
  }
  // Join

  printf("\n2.1 - Estamos fora do contexto paralelo...\n\n");

  printf("1.2 - Estamos fora do contexto paralelo...\n\n");

  // Fork
  #pragma omp parallel num_threads(4)
  {
    int num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    printf("Eu sou a Thread %d de um total de %d\n", thread_id, num_threads);
  }
  // Join

  printf("\n2.2 - Estamos fora do contexto paralelo...\n\n");

  return 0;
}

// =============================================================================
// FUNCTIONS
// =============================================================================

void set_portuguese(){
  setlocale(LC_ALL, "Portuguese");
}

void cabecalho(){
  printf("\n**************************************************");
  printf("\n*                                                *");
  printf("\n*                                                *");
  printf("\n* PROGRAMACAO PARALELA COM OPENMP - LUCCA PESSOA *");
  printf("\n*                                                *");
  printf("\n*                                                *");
  printf("\n**************************************************\n");
}
