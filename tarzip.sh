#!/bin/bash

nametar=FVE_DG_$(date +%Y%m%d_%H%M).tar.bz2
name7z=FVE_DG_$(date +%Y%m%d_%H%M).7z
# echo "$datename"
cd ..

if command -v pbzip2 &> /dev/null; then
    tar -cvf $nametar --exclude=Profiler --exclude=FVE/* --exclude=*/.ipynb_checkpoints --exclude=QuadData/*.pdf --exclude=Order_*/*.txt --exclude=RK?_Order_*/*.txt --use-compress-program="pbzip2 -9" FVE_DG
else
    tar -cvjf $nametar --exclude=Profiler --exclude=FVE/* --exclude=*/.ipynb_checkpoints --exclude=QuadData/*.pdf --exclude=Order_*/*.txt --exclude=RK?_Order_*/*.txt FVE_DG
fi

# 7zz a $name7z FVE
# cd FVE