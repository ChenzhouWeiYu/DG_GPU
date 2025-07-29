#!/bin/bash
#SBATCH --job-name=my_job           # 
#SBATCH --output=output_%j.log      # 
#SBATCH --error=error_%j.log        # 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1        
#SBATCH --time=24:00:00     
#SBATCH --partition=partMath  

#cd changgao/czwy/FVE_DG/examples/Eular2D/DoubleMach

#./DoubleMach 1 8 1 > stdout_8.txt 2>stderr_8.txt
#./DoubleMach 1 12 1 > stdout_12.txt 2>stderr_12.txt
#./DoubleMach 1 16 1 > stdout_16.txt 2>stderr_16.txt
#./DoubleMach 1 20 1 > stdout_20.txt 2>stderr_20.txt
#./DoubleMach 1 30 1 > stdout_30.txt 2>stderr_30.txt

./cylinder 2 4 32



#export OMP_NUM_THREADS=16
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close

#srun --exclusive --ntasks=1 ./Sedov 1 40 > stdout_1.txt 2>stderr_1.txt &
#srun --exclusive --ntasks=1 ./Sedov 2 20 > stdout_2.txt 2>stderr_2.txt &

wait
