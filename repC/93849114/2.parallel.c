#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#define N 25

/* Q1: Is the code printing what you expected? Is it executing        */
/*     in parallel? What is wrong with it?                            */
/* Q2: Add a directive to make its execution correct.                 */
/* Q3: What would happen if you remove the firstprivate clause        */
/*     in the task directive? And if you ALSO remove the firstprivate */
/*     clause in the parallel directive? Why are they redundant?      */
/* Q4: Why the program breaks when variable p is not firstprivate to  */
/*     the task?                                                      */
/* Q5: Why the firstprivate clause was not needed in 1.serial.c?      */

struct node {
   int data;
   int fibdata;
   int threadnum;
   struct node* next;
};

int fib(int n) {
   int x, y;
   if (n < 2) {
        return(1);
   } else {
      x = fib(n - 1);
      y = fib(n - 2);
      return (x + y);
   }
}

void processwork(struct node* p) 
{
   int n;
   n = p->data;
   p->fibdata += fib(n);
   p->threadnum = omp_get_thread_num();
}

struct node* init_list(int nelems) {
    int i;
    struct node *head, *p1, *p2;
    
    p1 = malloc(sizeof(struct node));
    head = p1;
    p1->data = 0;
    p1->fibdata = 0;
    p1->threadnum = 0;
    for (i=0; i<nelems-1; i++) {
       p2  = malloc(sizeof(struct node));
       p1->next = p2;
       p2->data = i+1;
       p2->fibdata = 0;
       p2->threadnum = 0;
       p1 = p2;
    }
    p1->next = NULL;
    return head;
}

int main(int argc, char *argv[]) {
     struct node *p, *temp, *head;
     
     printf("Staring computation of Fibonacci for numbers in linked list \n");
 
     p = init_list(N);
     head = p;

     #pragma omp parallel firstprivate(p) num_threads(4)
     #pragma omp single
     while (p != NULL) {
	   #pragma omp task //firstprivate(p)
	   processwork(p);
	   p = p->next;
     }

     printf("Finished computation of Fibonacci for numbers in linked list \n");
     p = head;
     while (p != NULL) {
        printf("%d: %d computed by thread %d \n", p->data, p->fibdata, p->threadnum);
        temp = p->next;
        free (p);
        p = temp;
     }  
     free (p);

     return 0;
}
