#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <iostream>
#include <stdio.h>
#define LEVEL 100

#include <curand.h>
#include <curand_kernel.h>

__device__
void printSolution(int *color , int *graph, int V) {
    printf("Solution Exists:"
        " Following are the assigned colors \n");
    for (int i = 0; i < V; i++)
        printf(" %d ", color[i]);
    printf("\n");
}

/* Function to check if the color can be safely assigned */
__device__
bool isSafe (int v, int *graph, int *color, int c, int V) {
    for (int i = 0; i < V; i++)
        if (graph[v*V + i] == 1 && c == color[i])
            return false;
    return true;
}

// __device__
// void graphColoringUtil(int graph[][100], int m, int color[], int v) {
//     if(found==false) {
//         if (v == V) {
//             printSolution(color,graph);
//             return;
//         }
//         for (int c = 1; c <= m; c++) {
//         /* Check if assignment of color c to v is fine*/
//             color[v] = c;
//             if (isSafe(v, graph, color, c)) {
//                 graphColoringUtil (graph, m, color, v+1);
//             }
//         }
//         return;
//     }
// }

__global__
void graphColoringUtilParallel(int *graph, int *m, int *color, int v, bool *found, int *V, int *temp, bool *flag, curandState_t state, unsigned int seed) {
    if (*flag) {
        // curandState_t state;
        /* we have to initialize the state */
        curand_init(seed, /* the seed controls the sequence of random values that are produced */
                0, /* the sequence number is only important with multiple cores */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);
        *flag = false;
    }
    if(*found==false) {
        // for (int i = 1; i <= *m; ++i) {
        // while (1) {
            if (v == *V) {
                printSolution(color,graph, *V);
                *found = true;
                return;
            }

            color[v] = curand(&state)%(*m) + 1;
            // color[v] = (*temp)%(*m) + 1;

            if (isSafe(v, graph, color, color[v], *V)) {
                if (v < LEVEL) {

                    int *tempColors = new int[100];
                    // cudaMallocManaged(&color, (*V)*sizeof(int));
                    for (int j = 0; j <= v; ++j) {
                        tempColors[j] = color[j];
                    }

                    graphColoringUtilParallel<<<1,4>>>(graph, m, tempColors, v+1, found, V, temp, flag, state, seed);
                    // cudaDeviceSynchronize();
                    delete [] tempColors;
                    // #pragma omp task firstprivate(v)
                    // {
                    //     int id = omp_get_thread_num();
                    //     printf("Thread assigned %d\n",id );
                    //     graphColoringUtilParallel(graph, m, tempColors, v+1);   // generate task of serial function
                    //     graphColoringUtilParallel<<<1,8>>>(graph, m, tempColors, *v+1, found, V);
                    // }
                }
                // else{
                    // #pragma omp taskwait
                    // graphColoringUtil(graph, m, color, v+1);
                // }
            }
        // }
        for (int i = 0; i < *V; i++)
            printf("thread id %d %d ", threadIdx.x, color[i] );
        printf("\n");
        return;
    }
}

void graphColoring(int *graph, int *m, int *V, bool *found) {
    // Initialize all color values as 0.
    int *color, *temp;//, *start; // = new int[V];
    cudaMallocManaged(&color, (*V)*sizeof(int));
    cudaMallocManaged(&temp, sizeof(int));

    // cudaMallocManaged(&start, sizeof(int));
    // *start = 0;
    for (int i = 0; i < *V; i++)
        color[i] = 0;

    // #pragma omp parallel shared(found)
    // {
    //     #pragma omp single
    //     {
    //         graphColoringUtilParallel(graph, m, color, 0 );
    //     }
    // }
    bool *flag;
    cudaMallocManaged(&flag, sizeof(bool));
    *flag = true;
    curandState_t state;
    graphColoringUtilParallel<<<1,1>>>(graph, m, color, 0, found, V, temp, flag, state, time(NULL));
    cudaDeviceSynchronize();

    cudaFree(color);
    cudaFree(temp);
}


int main() {
    srand(time(NULL));

    int _vertices, _colors;
    std::cout << "Enter number of vertices: ";
    std::cin >> _vertices;
    std::cout << "Enter number of colours: ";
    std::cin >> _colors;

    struct timeval  TimeValue_Start;
    struct timezone TimeZone_Start;

    struct timeval  TimeValue_Final;
    struct timezone TimeZone_Final;
    long   time_start, time_end;
    double  time_overhead;

    // Number of vertices, colors
    int *V, *m, *graph;
    bool *found;

    cudaMallocManaged(&V, sizeof(int));
    cudaMallocManaged(&m, sizeof(int));
    cudaMallocManaged(&graph, ((_vertices*2) + 1)*sizeof(int));
    cudaMallocManaged(&found, sizeof(bool));

    *V = _vertices;
    *m = _colors;
    *found = false;

    /* Example Graph
        (3)---(2)
         |   / |
         |  /  |
         | /   |
        (0)---(1)
        {{0, 1, 1, 1},
            {1, 0, 1, 0},
            {1, 1, 0, 1},
            {1, 0, 1, 0},
        };
    */

    for(int i=0;i<_vertices;i++) {
        for (int j=0;j<_vertices;j++) {
            if(i==j)
                graph[i*_vertices + j]=0;
            else {
                graph[i*_vertices + j] = rand()%2;
                graph[j*_vertices + i] = graph[i*_vertices + j];
            }
        }
    }


    printf("Adjacency Matrix\n");

    for(int i=0;i<_vertices;i++) {
        for (int j=0;j<_vertices;j++)
            printf("%d ", graph[i*_vertices + j]);
        printf("\n");
    }

    gettimeofday(&TimeValue_Start, &TimeZone_Start);

    graphColoring (graph, m, V, found);

    if(*found==false)
        printf("No solution exists\n");

    gettimeofday(&TimeValue_Final, &TimeZone_Final);

    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead = (time_end - time_start)/1000000.0;

    printf("\n Time in Seconds (T)  : %lf",time_overhead);

    cudaFree(V);
    cudaFree(m);
    cudaFree(found);
    cudaFree(graph);

    return 0;
}
