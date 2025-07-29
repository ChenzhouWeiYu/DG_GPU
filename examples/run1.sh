# g++ VortexBatch.cpp -o VortexBatch -fopenmp -mavx2 -mfma -O3 -g -std=c++17 

# #./VortexBatch 3 60 > logging_3_60_1
# #./VortexBatch 3 80 > logging_3_80_1

# for N in {5,10,15,20,25,30,40,60,80}; do  
#     ./VortexBatch 4 $N > logging_4_${N} 
# done
# #./VortexBatch 3 160 > logging_3_160

# for N in {5,10,15,20,25,30,40,60,80}; do  
#     ./VortexBatch 5 $N > logging_5_${N} 
# done


for N in {3,4,5,6,8};do ./PoiseuilleBatch 1 ${N} > Poiseuille/logging_1_${N};done
for N in {3,4,5,6,8};do ./PoiseuilleBatch 2 ${N} > Poiseuille/logging_2_${N};done
for N in {3,4,5,};do ./PoiseuilleBatch 3 ${N} > Poiseuille/logging_3_${N};done
for N in {10,12};do ./PoiseuilleBatch 1 ${N} > Poiseuille/logging_1_${N};done
for N in {10,12};do ./PoiseuilleBatch 2 ${N} > Poiseuille/logging_2_${N};done
for N in {6,8,10};do ./PoiseuilleBatch 3 ${N} > Poiseuille/logging_3_${N};done
./PoiseuilleBatch 4 6 > Poiseuille/logging_4_6
./PoiseuilleBatch 1 16 > Poiseuille/logging_1_16
./PoiseuilleBatch 2 16 > Poiseuille/logging_2_16
./PoiseuilleBatch 3 12 > Poiseuille/logging_3_12
./PoiseuilleBatch 4 8 > Poiseuille/logging_4_8
./PoiseuilleBatch 1 20 > Poiseuille/logging_1_20
./PoiseuilleBatch 2 20 > Poiseuille/logging_2_20
./PoiseuilleBatch 3 16 > Poiseuille/logging_3_16
./PoiseuilleBatch 4 12 > Poiseuille/logging_4_12