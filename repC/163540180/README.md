# Parallel_Computing
Some simple examples and advanced examples for Parallel Computing


# OpenMP
```
#include <omp.h>

omp_set_num_threads(nthread);
omp_get_thread_num();
omp_get_num_threads();


#pragma omp parallel

#pragma omp for

#pragma omp parallel for private(j) reduction(+: not_primes) schedule(dynamic)

#pragma omp parallel shared(t, t_old) private(i,j,iter) firstprivate(niter) reduction(max:dt)


```




# MPI
```
#include <stdio.h>
#include "mpi.h"

main(int argc, char** argv){

  int my_PE_num;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);

  printf("Hello from %d.\n", my_PE_num);

  MPI_Finalize();

}
```


```
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);

MPI_Status status;
MPI_Recv( &numbertoreceive, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

MPI_Send( &numbertosend, 1, MPI_INT, index, 10, MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD);

MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);
MPI_Comm_size(MPI_COMM_WORLD, &npes);


MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

MPI_Finalize();

```





#
