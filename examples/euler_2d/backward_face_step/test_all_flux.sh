#!/bin/bash

N=256
p=1
for FLUX in {LF,Roe,HLL,HLLC,RHLLC};do
    ./run ${p} ${N} ${FLUX} >/dev/null
    ./run ${p} ${N} ${FLUX}_WENO >/dev/null
done;
