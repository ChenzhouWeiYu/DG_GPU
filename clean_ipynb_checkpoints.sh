#!/bin/bash

for fp in $(find . -iname ".ipynb_checkpoints") ;do rm -r $fp;done
for fp in $(find src -iname ".ipynb_checkpoints") ;do rm -r $fp;done
for fp in $(find include -iname ".ipynb_checkpoints") ;do rm -r $fp;done
for fp in $(find build -iname ".ipynb_checkpoints") ;do rm -r $fp;done
for fp in $(find examples -iname ".ipynb_checkpoints") ;do rm -r $fp;done


