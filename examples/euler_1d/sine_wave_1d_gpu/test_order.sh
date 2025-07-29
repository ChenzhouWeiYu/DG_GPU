!/usr/bash

for p in {1,2,3}; do
    for N in {2,4,8,16}; do
        # ./run ${p} ${N} LFNOLimiter
        # ./run ${p} ${N} HLLNOLimiter
        # ./run ${p} ${N} HLLCNOLimiter
        ./run ${p} ${N} RoeNOLimiter
    done
done


