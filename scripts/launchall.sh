#!/bin/bash


# Define the array of values
memvalues=(10 30 60)
niter=300
nrep=10

# Define the script names
scripts=("testCuda.sh"  "testHip.sh"  "testKokkos.sh"  "testOmpLLVM.sh"  "testOmpV.sh"  "testOpenacc.sh"  "testPstlA.sh"  "testPstlV.sh"  "testSyclA.sh"  "testSyclI.sh")



# Loop through the values and scripts
for value in "${memvalues[@]}"; do
    for script in "${scripts[@]}"; do
        echo "Running $script with memvalue=$value, niter=$niter, nrep=$nrep"
        sbatch "$script" "$value" "$arg1" "$arg2"
    done
done


