# OpenMP

Breaks code into multiple threads
```c
#pragma omp parallel
{
	// parallel code goes here
}

```


## Variable Scoping
OpenMP has a *private-list* and *shared-list* in the Parallel Region. General default is shared for Parallel Region, firstprivate for Tasks

### Private
Set `x` as a variable that is shared between all threads within the block
```c
int x;
#pragma omp task shared(x)
{
	// parallel code goes here
}
```


Set `b` as  private variable, where it has a copy for every thread (i.e. is just like `c`)
```c
int a, b;
#pragma omp parallel shared(a) private(b)
{
	int c;
}
```

- Non-static variables local to Parallel Regions are private
- Static variables are shared by default.
- Loop control variables on for-constructs are private
- `firstprivate` = Initialization with Master‘s value
- `lastprivate` = Value of last loop iteration is written back to Master


## The Barrier Construct
Threads wait until all threads of the current Team have reached the barrier
```c
#pragma omp barrier
```


## Internal Control Variables
The following variables are (some) of the variables available through OpenMP
```c
OMP_NUM_THREADS
OMP_NUM_TEAMS
```
They can be accessed using standard getters:
```c
omp_get_thread_num();
omp_get_num_threads();
omp_get_num_teams();
omp_get_wtime();
```

## Loop Parallelisation

This code has no loop dependencies, so it is paralleliseable
```c
#pragma omp parallel for
for ( i=0 ; i<100 ; i++ )
{
	a[i] = b[i] + c[i];
}
```

In this code, the calculation for `a[i]` depends on `a[i-1]` which means it cannot be parallelised in the case that `a[i-1]` has not been executed before `a[i]`
```c
#pragma omp parallel for
for ( i=1 ; i<100 ; i++ )
{
	a[i] = a[i] + a[i-1];
}
```

### Reductions
The reduction(+:a) means to sum all of the a values shared across the threads to
```c
int a = 0;

#pragma omp parallel
#pragma omp for reduction(+:a)
for (int i=0; i<100; i++)
{
	a+=i;
}
```

Possible reduction operations (with initialisation value):
```
+ (0),
* (1),
- (0),
& (~0),
| (0),
&& (1),
|| (0),
^ (0),
min (least number),
max (largest number)
```

## Synchronisation
A Critical Region is executed by all threads, but by only one thread simultaneously (Mutual Exclusion).
```c
#pragma omp critical (name)
{
	... structured block ...
}
```

```c
#pragma omp task wait
```

## The Single Construct
The single construct specifies that the enclosed structured block is executed by only on thread of the team.
```c
#pragma omp single[clause]
... structured block ...
```

## The Master Construct
The master construct specifies that the enclosed structured block is executed only by the master thread of a team. Note: The master construct is no worksharing construct and does not contain an implicit barrier at the end.
```c
#pragma omp master[clause]
... structured block ...
```

## The Sections Construct
The sections construct contains a set of structured blocks that are to be distributed among and executed by the team of threads.
```c
#pragma omp sections [clause]
{
	#pragma omp section
	... structured block ...
	#pragma omp section
	... structured block ...
...
}
```
Could be used for parallising multiple options (left/right branches of a binary tree)
```c
#pragma omp parallel sections
{
	#pragma omp section
	if (tree->left) 
		traverse(tree->left);
	
	#pragma omp section
	if (tree->right)
		traverse(tree->right);
} // end omp parallel
```

## The Ordered Construct
Allows a structured block to be executed within a parallel loop in sequential
order. In addition, an `ordered` clause has to be added to the `for` construct which any
ordered construct may occur (can be used to enfore order on printing data for example).  May also help to determine whether there is a data race.
```c
#pragma omp parallel for ordered
for (i=0 ; i<10 ; i++)
{
	...
	#pragma omp ordered
	{
		...
	}
	...
}
```

## Scheduling
### Static Scheduling
Breaks the work between all the threads before they are started (default setting
```c
#pragma omp ... schedule(static)
```
 Below are examples taken from ![Jakas Corner](http://jakascorner.com/blog/2016/06/omp-for-scheduling.html) who has some great notes.
```
schedule(static):      
****************                                                
                ****************                                
                                ****************                
                                                ****************
```
```
schedule(static, 4):   
****            ****            ****            ****            
    ****            ****            ****            ****        
        ****            ****            ****            ****    
            ****            ****            ****            ****
```
```
schedule(static, 8):   
********                        ********                        
        ********                        ********                
                ********                        ********        
                        ********                        ********
```

### Dynamic Scheduling
Provides only one (by default) process to each thread, and once complete they request a new task.  This adds lots of communication overhead but allows processes to be scheduled very dynamically
```c
#pragma omp ... schedule(dynamic)
```
```
schedule(dynamic):     				same as (dyanmic, 1)
*   ** **  * * *  *      *  *    **   *  *  * *       *  *   *  
  *       *     *    * *     * *   *    *        * *   *    *   
 *       *    *     * *   *   *     *  *       *  *  *  *  *   *
   *  *     *    * *    *  *    *    *    ** *  *   *     *   * 
```
```
schedule(dynamic, 4):  
            ****                    ****                    ****
****            ****    ****            ****        ****        
    ****            ****    ****            ****        ****    
        ****                    ****            ****            
```
```
schedule(dynamic, 8):  
                ********                                ********
                        ********        ********                
********                        ********        ********        
        ********                                                
```

### Guided
Sets a package size of n operations, where n is defined by OpenMP, and allocates each thread packages of n operations as they become available
```c
#pragma omp ... schedule(guided)
```
```
schedule(guided):      
                            *********                        *  
                ************                     *******  ***   
                                     *******                   *
****************                            *****       **    * 
```
```
schedule(guided, 2):   
                ************                     ****     **    
                                     *******         ***    **  
                            *********                           
****************                            *****       **    **
```
```
schedule(guided, 4):   
                                     *******                    
                ************                     ****    ****   
                            *********                           
****************                            *****    ****    ***
```
```
schedule(guided, 8):   
                ************                 ********        ***
****************                                                
                                     ********                   
                            *********                ********
```
<br/>

# MPI
http://www.open-mpi.org/

## Hello World Example
```c
#include <stdio.h>
#include <stdlib.h>

/*
* mpicc -std=c99 hello.c -o hello
* mpiexec -np 3 hello
*/
int main(int argc, char* argv[])
{
	int procRank, procCount;
	
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &procCount);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
	
	printf("Start[%d]/[%d]: Hello World\n",procRank,procCount);
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
```
## Data Types
MPI Type     | C Type
-------------|-----------------
MPI_CHAR     | signed char
MPI_INT      | signed int
MPI_LONG     | signed long int
MPI_UNSIGNED | unsigned int
MPI_FLOAT    | float
MPI_DOUBLE   | double

## Predefined Communicators
MPI communication always takes place within a communicator, a communicator a group is used to describe the participants in a communication "universe".  This is properly defined after `MPI_Init(...)` has been called
```c
MPI_COMM_WORLD //includes all of the started processes 

MPI_COMM_SELF //includes only the process itself
```

### Size
Number of processes within a group, can be determined using:
```c
int MPI_Comm_size(MPI_COMM_WORLD, &procCount);
```

### Rank
Identifies processes within a group, can be determined using:
```c
int MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
```

## Point to Point Communication
### Send
```c
int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```
- **buf**: starting address of the message
- **count**: number of elements
- **datatype**: type of each element
- **dest**: rank of destination in communicator comm
- **tag**: message identification
- **comm**: communicator

Example:
```c
MPI_Send(&message, 1, MPI_INT, 1, tagSend, MPI_COMM_WORLD);
```

### Recieve
```c
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int src , int tag, MPI_Comm comm, MPI_Status *status)
```
In addition `MPI_Recv` has:
- **status**: envelope information (message information)
```c
MPI_Recv(message, length, MPI_INT, i, tagSend, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

### Sources and Tags
It is possible to receive messages from any other process and with any message tag by using `MPI_ANY_SOURCE` and `MPI_ANY_TAG`. Actual source and tag are included in the status variable.

```c
int src = status.MPI_SOURCE;
int tag = status.MPI_TAG;
MPI_Get_count(*status, mpi_datatype, *count)  // element count of data
```

## Blocking and Non-Blocking 
### Blocking
Holds the program until the action (read, write, send, recieve, etc.) has completed. For example, ordering a package and then waiting inside all day until the delivery arrives because there is nowhere to store your package. `MPI_Send` and `MPI_Recv` are both blocking functions, they hold up execution of the next line of code until the action is completed.
 
### Non-blocking
Frees the program to return and check whether the action ahs completed.  To continue the analogy, delivery to a pickup point to allow you to check whenever you line so you can do stuff with your day.

To track communication requests an integer request handle is provided by the MPI system
```c
MPI_Request req;
MPI_Issend(&message, 1, MPI_INT, 1, tagSend, MPI_COMM_WORLD, &req)
```

In non-blocking send-variants we need to check for the communication's completion either by waiting for the communication to be complete:
```c
int MPI_Wait(MPI_Request *req, MPI_Status *status)
```
or looping and testing over and over to see if the comminication is completed
```c
MPI_Test(&req,&flag,MPI_STATUS_IGNORE);
```

example:
```c
message = 42;
MPI_Request req;
MPI_Issend(&message, 1, MPI_INT, 1, tagSend, MPI_COMM_WORLD, &req);

int flag = 0;
while (1)
{
	MPI_Test(&req,&flag,MPI_STATUS_IGNORE);
	if (1 == flag)
		break;
	printf("wait ...\n");
	sleep(1);
}
printf("Message sent \n");
```

## Synchronous and Asynchronous (Buffered)
### Synchronous Communication
The send call will only start when the destinantion is ready to recieve (like a telephone call).  The send call will only complete after the message has been completely transmitted to the recieving process and it acknowledges the message.

### Asynchronous Communication
The send can complete even if the recieve is not ready (like an SMS message). This is stored in a buffer of messages that the reciever can access when ready. In MPI this can be done with `MPI_Bsend`

Mode        	|Standard    		|Synchronous    |Asyncronous (Buffered)
----------------|-----------------------|---------------|-----------------------
Blocking	|MPI_Send/MPI_Recv	|MPI_Ssend	|MPI_Bsend
Non-blocking	|MPI_Isend/MPI_Irecv	|MPI_Issend	|MPI_Ibsend

The `MPI_Send` can be both synchronous and asynchronous depending on the implementation and message size. This is why `MPI_Ssend` can force synchronos and `MPI_Bsend` can force asynchronous. 
