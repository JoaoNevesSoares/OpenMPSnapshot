#!/bin/bash

####################################################################
#        ---- MMCK demo: case 2 - rectangular matrices ----        #
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

# matrix dimensions: rectangular matrices (case 2)
dims_rect_mn=10000
dims_rect_k=(32 64 128 256)

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

for mn in "${dims_rect_mn[@]}"
	do
		for k in "${dims_rect_k[@]}"
			do
				rm -f ./data/A.dat ./data/B.dat ./data/C.dat
				rm -f ./data/RES-mpi.dat ./data/RES-mixed.dat ./data/RES-omp.dat
				rm -f ./data/expected.dat

				echo "================================================================="
				echo "> Scaling number of processes required, (m,k,n)=($mn,$k,$mn)"
				echo "-----------------------------------------------------------------"
				echo "Limits: ($procs_min, $procs_max)"
				npscal=$(./scripts/compute-scalable-nprocs.sh "$procs_min" "$procs_max" "$mn"*"$k")
				echo "$npscal processes are required."
				echo "Done."

				# Matrices generation
				echo -e
				echo "================================================================="
				echo "> Matrix generation: (m,k,n)=($mn,$k,$mn), $npscal procs"
				echo "-----------------------------------------------------------------"
				mpirun	-np "$npscal" ./tools/generator/build/GENERATOR "$mn" "$k" 3 "A.dat"
				mpirun	-np "$npscal" ./tools/generator/build/GENERATOR "$k" "$mn" 3 "B.dat"
				mpirun	-np "$npscal" ./tools/generator/build/GENERATOR "$mn" "$mn" 3 "C.dat"
				for np in "${procs[@]}"
					do
						for block in "${dims_block[@]}"
							do
								for i in "${iterations[@]}"
									do
										echo -e
										echo "================================================================="
										echo "> MPI: (m,k,n)=($mn,$k,$mn), $np procs, $block x $block blocks, iteration $i"
										echo "-----------------------------------------------------------------"
										mpirun	-np "$np" ./demo-mpi/build/MMCK-MPI-DEMO "$mn" "$k" "$mn" "$block" "$block" "A.dat" "B.dat" "C.dat"
										mpirun	-np "$npscal" ./tools/validator/build/VALIDATOR "$mn" "$k" "$mn" "A.dat" "B.dat" "C.dat" "RES-mpi.dat" "stats-mpi.csv" 5.0 "$npscal"
									done
								for nt in "${procs[@]}"
									do
										for i in "${iterations[@]}"
											do
												if [  $((np*nt)) -le 20 ];
                      								then
														echo -e
														echo "================================================================="
														echo "> MIXED: (m,k,n)=($mn,$k,$mn), $np procs x $nt threads, $block x $block blocks, iteration $i"
														echo "-----------------------------------------------------------------"
														mpirun	-np "$np" ./demo-mixed/build/MMCK-MIXED-DEMO "$mn" "$k" "$mn" "$block" "$block" "$nt" "A.dat" "B.dat" "C.dat"
														mpirun	-np "$npscal" ./tools/validator/build/VALIDATOR "$mn" "$k" "$mn" "A.dat" "B.dat" "C.dat" "RES-mixed.dat" "stats-mixed.csv" 5.0 "$npscal"
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
										echo "> OMP: (m,k,n)=($mn,$k,$mn), $np threads, pragma = $pragma, iteration $i"
										echo "-----------------------------------------------------------------"
										./demo-omp/build/MMCK-OMP-DEMO "$np" "$mn" "$k" "$mn" "$pragma" "A.dat" "B.dat" "C.dat"
										mpirun	-np "$npscal" ./tools/validator/build/VALIDATOR "$mn" "$k" "$mn" "A.dat" "B.dat" "C.dat" "RES-omp.dat" "stats-omp.csv" 5.0 "$npscal"
									done
							done
					done
			done
	done

