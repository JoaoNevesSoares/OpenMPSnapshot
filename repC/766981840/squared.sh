#!/bin/bash

####################################################################
#          ---- MMCK demo: case 1 - squared matrices ----          #
####################################################################

#*=========================*
#| Environment operations  |
#*-------------------------*/

export HWLOC_HIDE_ERRORS=2
module load -s mpi

#*=========================*
#| Processors no. limits   |
#*-------------------------*/

# no. of available processors
procs_min=2
procs_max=20

#*=========================*
#| Matrices parameters     |
#*-------------------------*/

# matrix dimensions: squared matrices (case 1)
dims_squa_mkn=($(seq 250 250 10000))

# blocks dimensions (MPI, mixed)
dims_block=(4 8 16 32)

# numbers of processes/threads
procs=(2 4 8 12 16 20)

# "pragmas" (OMP version; see report)
pragmas=(0 1 2 3)

# no. of iterations per each computation
iterations=($(seq 1 1 5))

#*=========================*
#| The actual computations |
#*-------------------------*/

for mkn in "${dims_squa_mkn[@]}"
	do
		rm -f ./data/A.dat ./data/B.dat ./data/C.dat
		rm -f ./data/RES-mpi.dat ./data/RES-mixed.dat ./data/RES-omp.dat
		rm -f ./data/expected.dat

		echo "================================================================="
		echo "> Scaling number of processes required, (m,k,n)=($mkn,$mkn,$mkn)"
		echo "-----------------------------------------------------------------"
		echo "Limits: ($procs_min, $procs_max)"
		npscal=$(./scripts/compute-scalable-nprocs.sh "$procs_min" "$procs_max" "$mkn"*"$mkn")
		echo "$npscal processes are required."
		echo "Done."

		# Matrices generation
		echo -e
		echo "================================================================="
		echo "> Matrix generation: (m,k,n)=($mkn,$mkn,$mkn), $npscal procs"
		echo "-----------------------------------------------------------------"
		mpirun	-np "$npscal" ./tools/generator/build/GENERATOR "$mkn" "$mkn" 3 "A.dat"
		mpirun	-np "$npscal" ./tools/generator/build/GENERATOR "$mkn" "$mkn" 3 "B.dat"
		mpirun	-np "$npscal" ./tools/generator/build/GENERATOR "$mkn" "$mkn" 3 "C.dat"
		for np in "${procs[@]}"
			do
				for block in "${dims_block[@]}"
					do
						for i in "${iterations[@]}"
							do
								echo -e
								echo "================================================================="
								echo "> MPI: (m,k,n)=($mkn,$mkn,$mkn), $np procs, $block x $block blocks, iteration $i"
								echo "-----------------------------------------------------------------"
								mpirun	-np "$np" ./demo-mpi/build/MMCK-MPI-DEMO "$mkn" "$mkn" "$mkn" "$block" "$block" "A.dat" "B.dat" "C.dat"
								mpirun	-np "$npscal" ./tools/validator/build/VALIDATOR "$mkn" "$mkn" "$mkn" "A.dat" "B.dat" "C.dat" "RES-mpi.dat" "stats-mpi.csv" 5.0 "$npscal"
							done
						for nt in "${procs[@]}"
							do
								for i in "${iterations[@]}"
									do
										if [  $((np*nt)) -le 20 ];
											then
											echo -e
											echo "================================================================="
											echo "> MIXED: (m,k,n)=($mkn,$mkn,$mkn), $np procs x $nt threads, $block x $block blocks, iteration $i"
											echo "-----------------------------------------------------------------"
											mpirun	-np "$np" ./demo-mixed/build/MMCK-MIXED-DEMO "$mkn" "$mkn" "$mkn" "$block" "$block" "$nt" "A.dat" "B.dat" "C.dat"
											mpirun	-np "$npscal" ./tools/validator/build/VALIDATOR "$mkn" "$mkn" "$mkn" "A.dat" "B.dat" "C.dat" "RES-mixed.dat" "stats-mixed.csv" 5.0 "$npscal"
											fi
									done
							done
					done
				for pragma in "${pragmas[@]}"
					do
						for i in "${iterations[@]}"
							do
								echo -e
								echo "================================================================="
								echo "> OMP: (m,k,n)=($mkn,$mkn,$mkn), $np threads, pragma = $pragma, iteration $i"
								echo "-----------------------------------------------------------------"
								./demo-omp/build/MMCK-OMP-DEMO "$np" "$mkn" "$mkn" "$mkn" "$pragma" "A.dat" "B.dat" "C.dat"
								mpirun	-np "$npscal" ./tools/validator/build/VALIDATOR "$mkn" "$mkn" "$mkn" "A.dat" "B.dat" "C.dat" "RES-omp.dat" "stats-omp.csv" 5.0 "$npscal"
							done
					done
			done
	done
