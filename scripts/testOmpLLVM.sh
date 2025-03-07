#!/bin/bash --login
 
 
#SBATCH --account=
#SBATCH --partition=
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --gpus-per-node=1 
#SBATCH --time=04:00:00
#SBATCH --exclusive

## load necessary modules and set enviromental variables

##-----------------------------------------------------------------------------------------------------------------------------------------------
mkdir -p logs_test

for ((i = 1; i <= $3; i++)); do
../bin/GaiaGsrParSimOMPGpu_LLVM.x -memGlobal $1 -IDtest 0 -itnlimit $2 > logs_test/log.1GPU_$1_$2_llvmomp$i
done

