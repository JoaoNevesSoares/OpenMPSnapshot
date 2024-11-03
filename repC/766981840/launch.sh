#!/bin/bash

####################################################################
#                   ---- MMCK demo script ----                     #
# This script builds all the MMCK versions and, for each of them,  #
# starts the "launcher.c" programs, that are used to compute       #
# some performance metrics.                                        #
####################################################################

echo	-e
echo 	"#=================================#"
echo 	"#    *** MMCK demo program ***    #"
echo 	"#---------------------------------#"

#*=========================*
#| Environment operations  |
#*-------------------------*/

export HWLOC_HIDE_ERRORS=2
module load -s mpi

#*=========================*
#| Filesystem operations   |
#*-------------------------*/

# Prepare the folder that will contain the generated matrices, if not already existent
mkdir -p data
rm -rf data/*

# Output files headers
echo "m,k,n,np,topo_size,supern,mb,nb,time_parall,gflops,time_seq,max_error_perc">data/stats-mpi.csv
echo "m,k,n,np,topo_size,supern,nt,np*nt,mb,nb,time_parall,gflops,time_seq,max_error_perc">data/stats-mixed.csv
echo "m,k,n,nt,pragma,time_parall,gflops,time_seq,max_error_perc">data/stats-omp.csv

#*=========================*
#| Tools compilation       |
#*-------------------------*/

# Compile the demo utility tools
# (library structure is identical)
tools=(generator validator)

for tool in "${tools[@]}"
	do
		./scripts/compile-tool.sh "$tool"
done

#*=========================*
#| Versions compilation    |
#*-------------------------*/

# Compile the different MMCK implementations and the matrix generator
# (library structure is identical)

versions=(mpi mixed omp)

for version in "${versions[@]}"
	do
		./scripts/compile-version.sh "$version"
done

#*=========================*
#| Start MMCK demos        |
#*-------------------------*/

./scripts/squared.sh
./scripts/rectangular.sh

