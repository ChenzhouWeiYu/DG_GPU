!/usr/bash

for p in {1,2,3,4}; do
    for N in {2,4,8,16}; do
        ./run ${p} ${N} LF
        ./run ${p} ${N} HLL
        ./run ${p} ${N} HLLC
        # ./run ${p} ${N} RoeNOLimiter
    done
done


