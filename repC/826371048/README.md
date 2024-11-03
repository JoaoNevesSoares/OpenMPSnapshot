# Mandelbrot Set - Parallel Task Decomposition Implementation and Analysis
This project was developed as part of the PAR (Parallelism) course at the FIB - UPC Barcelona. The focus is on the implementation and performance analysis of iterative and recursive task decomposition techniques using OpenMP for the computation of the Mandelbrot set. <br><br>

## Authors
- Arnau Claramunt
- David García

[![GitHub followers](https://img.shields.io/github/followers/ArnauCS03?label=ArnauCS03)](https://github.com/ArnauCS03) &nbsp;&nbsp; [![GitHub followers](https://img.shields.io/github/followers/dgarevalo?label=dgarevalo)](https://github.com/dgarevalo) <br><br>

---
>[!NOTE]  
>The codes are in the `src` folder, the statment is called *lab_practice_PAR.pdf* and our results are presented in *Lab4_report_PAR*. <br><br>

## Iterative task decomposition

The **iterative task decomposition** approach involves dividing the Mandelbrot set computation into smaller tasks, which are then processed in parallel. The main goal is to exploit parallelism while handling dependencies and data sharing to avoid concurrency issues. To improve efficiency, strategies like finer grain tasks were employed. This involves breaking down the computation into even smaller, more manageable tasks that can be distributed across multiple threads.  <br><br>

Key Modifications:

- Implemented in *mandel-omp-iter-finer.c* *mandel-omp-iter-v1.c*
- Used `#pragma omp task` for task creation within nested loops.
- Applied **atomic** and **critical** operations to avoid data race conditions.  <br><br>

Performance Analysis:

- Model Factors Analysis: Evaluated the efficiency and overhead of the parallel implementation. Identified and addressed bottlenecks.
- Paraver Analysis: Used Paraver to visualize and analyze the execution traces for 16 threads.
- Strong Scalability: Conducted scalability tests to determine how the performance scales with an increasing number of threads.  <br><br>

## Recursive Task Decomposition

The **recursive task decomposition** approach uses a divide-and-conquer strategy to break down the computation into smaller sub-problems. Each sub-problem is solved recursively, leveraging parallel tasks where possible. This method utilizes leaf and tree strategies to optimize task execution and load balancing. In the leaf strategy, tasks are only created at the lowest level of recursion, reducing overhead. The tree strategy, on the other hand, creates tasks at multiple levels of recursion, improving load distribution but potentially increasing overhead.

Key Modifications:

- Implemented in *mandel-omp-rec-leaf.c* *mandel-omp-rec-tree.c*
- Addressed dependencies and data sharing to prevent concurrency issues.

Performance Analysis:

- Model Factors Analysis: Similar to the iterative approach, evaluated the efficiency and identified bottlenecks in the recursive implementation.
- Paraver Analysis: Analyzed execution traces with Paraver for 16 threads to understand task distribution and synchronization.
- Strong Scalability: Tested how well the implementation scales with an increasing number of threads. <br><br>

## Results and Comparison

The results show that both the iterative and recursive implementations achieve significant speedups compared to the sequential versions. However, the recursive task decomposition demonstrated better scalability and efficiency, making it the preferred approach for this problem.  <br><br>

### Summary of Execution Times:

Execution times were recorded for different thread counts (1, 4, 8, 12, 16, 20).
The recursive implementation consistently outperformed the iterative one in terms of scalability and overall execution time.  <br><br>

## How to Run the Code

Compile the Code:
```
make
```

### Run or submit the Parallel Code: (We used the Boada Supercomputer for the computation)
```
./mandel-seq-iter [-o -h -d -i maxiter -c x0 y0 -s size]
sbatch submit-omp.sh mandel-omp-iter 20
```

<br><br>

## Conclusion

This project successfully demonstrates the implementation and analysis of parallel task decomposition techniques for the Mandelbrot set computation. The recursive task decomposition, in particular, showed superior performance and scalability, making it a valuable approach for parallel computations in similar applications.
<br><br>

---

## Screenshots

<p align="center">
  <img src="https://github.com/ArnauCS03/mandelbrot-set-omp-parallelization/assets/95536223/1bdccf91-75ef-4561-bc23-7514a540d223" width="400" height="400"/> 
</p>

![Screenshot from 2024-07-09 17-56-05](https://github.com/ArnauCS03/mandelbrot-set-omp-parallelization/assets/95536223/7b2aafa9-034f-4342-b7cc-8a5237564362)

<p align="center">
   <img src="https://github.com/ArnauCS03/mandelbrot-set-omp-parallelization/assets/95536223/2d1dd318-4fc4-45a2-8e7f-17c3c1d879d4" width="400" height="500"/> 
</p>

<br><br>

