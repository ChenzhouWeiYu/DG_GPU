#!/bin/bash

N=20
p=2
for FLUX in {LF,Roe,HLL,HLLC,RHLLC};do
    ./run ${p} ${N} ${FLUX} >/dev/null
    ./run ${p} ${N} ${FLUX}_WENO >/dev/null
done;
