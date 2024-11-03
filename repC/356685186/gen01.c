/*
    Description:
        This program implements my Genetic Algorithm method of solving the "N-Queens Problem"
        

    Author:
        Georgios Evangelou (1046900)
        Year: 5
        Parallel Programming in Machine Learning Problems
        Electrical and Computer Engineering Department, University of Patras
    
    System Specifications:
        CPU: AMD Ryzen 2600  (6 cores/12 threads,  @3.8 GHz,  6786.23 bogomips)
        GPU: Nvidia GTX 1050 (dual-fan, overclocked)
        RAM: 8GB (dual-channel, @2666 MHz)
        
    Version Notes:
        Compiles/Runs/Debugs with: gcc gen01.c -o gen01 -lm -fopt-info -pg && time ./gen01 && gprof ./gen01
        Sometimes it functions correctly and other times it cannot zero the utility function 
*/	


// ****************************************************************************************************************    
//#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Apply O3 and extra optimizations
//#pragma GCC option("arch=native","tune=native","no-zero-upper") //Adapt to the current system
//#pragma GCC target("avx")  //Enable AVX


// ****************************************************************************************************************    
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdbool.h"
#include "time.h"


// ****************************************************************************************************************    
#define N 5        //Number of queens
#define GENES 4    //Number of genes (must by even)



/**
 * Initializes positional array given
 */
void GeneInitialization(int genes[GENES][N]) {
    for (int i=0; i<GENES; i++) {
        for (int j=0; j<N; j++) {
            genes[i][j] = N*((float)rand())/((float)RAND_MAX);
        }
    }
}


/**
 * Prints a map of the queens until the M-th positioned queen
 */
void Map(int posY[N], int M, int g) {
    //for (int i=0; i<M; i++) printf("~~~~", i); 
    printf("\n========================\n"); 
    printf("------- GENE %d -------\n", g);
    printf("========================\n");
    for (int i=0; i<N; i++) printf("---"); printf("---\n##|");
    for (int i=0; i<N; i++) printf("%2d ", i+1); printf("\n---");
    for (int i=0; i<N; i++) printf("---"); printf("\n");
    
    for (int y=0; y<N; y++) {
        printf("%2d| ", y+1);
        for (int x=0; x<N; x++) {
            bool flag = false;
            for (int i=0; i<M; i++) {
                if (i==x && posY[i]==y) {
                    flag = true;
                }
            }
            if (flag) printf("Q");
            else printf("~");
            printf("  ");
        }
        printf("\n");
    }
    for (int i=0; i<N; i++) printf("---"); printf("---\n");
}


/**
 * Checks if a position is safe
 */
bool isSafeFromPrevious(int posY[N], int x, int y) {
    int currentQueen = x;
    for (int oldQueen=0; oldQueen<currentQueen; oldQueen++) {
        //printf("      Checking %d %d and %d %d \n",posX[q],posY[q],x,y);
        if (oldQueen==x || posY[oldQueen]==y) return false; //If row/column is endangered
        else if (y==posY[oldQueen]+(currentQueen-oldQueen) || y==posY[oldQueen]-(currentQueen-oldQueen)) return false; //If diagonal is endangered
    }
    return true;
}


/**
 * Finds the number collisions between the queens
 * TODO: HAS WRONG OUTPUT
 */ 
int UtilityFunction(int posY[N]) {
    int collisions = 0;
    for (int crnt=1; crnt<N; crnt++) {
        for (int old=0; old<crnt; old++) {
            if (old==crnt || posY[old]==posY[crnt]) collisions++; //If row/column is endangered
            else if (posY[crnt]==posY[old]+(crnt-old) || posY[crnt]==posY[old]-(crnt-old)) collisions++; //If diagonal is endangered
        }
    }
    return collisions;
}


/**
 * Takes two parent genes and produces two child genes
 */ 
void CrossoverFunction(int gene1[N], int gene2[N]) {
    for (int i=1; i<N; i++) {
        if (abs(gene1[i-1]-gene1[i])<2 || abs(gene2[i-1]-gene2[i])<2) {
            int temp = gene1[i];
            gene1[i] = gene2[i];
            gene2[i] = temp;
        }
    }
    return;
}


/**
 * Produces a random integer in the range [mini,maxi]
 */
int RandomInteger(int mini, int maxi) {
    int gap = maxi-mini;
    int randomInGap = (int) (gap * ((float)rand())/((float)RAND_MAX) ); //[0,gap]
    return mini + randomInGap; //[mini,mini+gap]==[mini,maxi]
}



/**
 * Takes a gene and mutates it
 */ 
void MutationFunction(int gene[N]) {
    bool missingFromGene[N] = {true};

    // Mark all values missing from the gene, so they can be used to replace duplicates
    for (int i=0; i<N; i++) {
        missingFromGene[gene[i]] = false;
    }
    
    // Find duplicates and replace them with non-used values
    for (int i=0; i<N; i++) {
        for (int j=0; j<i-1; j++) {
            if (gene[i]==gene[j]) {
                for (int k=0; k<N; k++){
                    if (missingFromGene[k]) {
                        gene[i] = k;
                        missingFromGene[k] = false;
                    }
                }
            }
        }
    }

    // Performs the actual swapping
    
    int barrier = RandomInteger(1,N-3); // [1, N-3]
    int swapA = RandomInteger(0,barrier);   // [0,barrier]
    int swapB = RandomInteger(barrier+1,N-1); // [barrier+1,N-1]
    if ((swapA<0 || swapA>=N)||(swapB<0 || swapB>=N)||(swapA>=swapB)) {printf("\nSWAP ERROR: barrier=%d, swapA=%d, swapB=%d\n", barrier, swapA, swapB);exit(1);    }
    
    if (RandomInteger(1,10)!=1) {
        int temp = gene[swapA];
        gene[swapA] = gene[swapB];
        gene[swapB] = temp;
    } else {
        //printf("last first\n");
        int temp = gene[0];
        gene[0] = gene[N-1];
        gene[N-1] = temp;
    }
    //printf("Swappings made: %d and %d\n", swapA, swapB);
    return;
}


/**
 * Breeds next generation
 */ 
void BreedGeneration(int genes[GENES][N], int utilityValues[GENES]) {
    int index1 = -1, index2 = -1;
    float maximum_value = INFINITY;
    float value1 = maximum_value, value2 = maximum_value;
    int genesNew[GENES][N];

    // For all pairs of genes to create
    for (int i=0; i<GENES; i+=2) {
        
        //...access all current genes and semi-randomly find two low-value parents
        for (int j=0; j<GENES; j++) {
            float value = (float) (10000 + rand()*utilityValues[j] );
            //float value = utilityValues[j];
            if (value<value1) {
                value2 = value1;
                index2 = index1;
                value1 = value;
                index1 = j;
            } else if (value<value2) {
                value2 = value;
                index2 = j;
            }
        }

        if (value1>=maximum_value || value2>=maximum_value) {
            printf("\nVALUE ERROR: value1=%f, value2=%f\n", value1, value2);
            printf("index1=%d, index2=%d\n", index1, index2);
            for (int kappa=0; kappa<GENES; kappa++) printf(">> Utility value %d: %d\n",kappa, utilityValues[kappa]);
            exit(1);
        }

        //...then copy the parents to the new array
        for (int k=0; k<N; k++) {
            genesNew[i][k]   = genes[index1][k];
            genesNew[i+1][k] = genes[index2][k];
        }

        //...breed and mutate their children
        CrossoverFunction(genesNew[i], genesNew[i+1]);
        MutationFunction(genesNew[i]);
        MutationFunction(genesNew[i+1]);
    }

    // Finally copy the new genes into the old ones
    for (int i=0; i<GENES; i++) {
        for (int j=0; j<N; j++) {
            genes[i][j] = genesNew[i][j];
        }
    }    
}


// Calculate and store all current genes utility values
int CalculateAllUtilityValues(int genes[GENES][N], int utilityValues[GENES]) {
    int bestUtilityValueFoundAt = 0;
    for (int i=0; i<GENES; i++) {
        if ((utilityValues[i]=UtilityFunction(genes[i])) < utilityValues[bestUtilityValueFoundAt]) {
            bestUtilityValueFoundAt = i;
        }
    }
    return bestUtilityValueFoundAt;
}


/**
 * The main program
 */
int main() {
    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    printf("This program implements my Genetic Algorithm method of solving the \"N-Queens Problem\".\n");
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    __time_t now = time(0);
    srand(now);
    int genes[GENES][N];
    int utilityValues[GENES];
    
    GeneInitialization(genes);

    //while(1) printf("%d\n", RandomInteger(0,4));
    //for (int i=0; i<GENES; i++) Map(genes[i], N, i);

    printf("Now solving the problem. Please wait...\n");
    
    int generation = 0, bestGene = -1;
    while(++generation) {
        bestGene = CalculateAllUtilityValues(genes, utilityValues);
        if (generation%100==0) printf("\rGeneration: %3d  Best value: %2d", generation, utilityValues[bestGene]);

        if (utilityValues[bestGene]==0) break;
    
        BreedGeneration(genes, utilityValues);
        
    }

    Map(genes[bestGene], N, bestGene); 
    bool allSafe = true;
    for (int i=0; i<N-1; i++) {
        if (!isSafeFromPrevious(genes[bestGene], i, genes[bestGene][i])) {
            printf("WRONG SOLUTION!!\n");
            allSafe = false;
        }
    }
    if (allSafe) printf("The solution was validated and is accepted.\n");
}

