#!/bin/bash


export CUDA_VISIBLE_DEVICES=""
source ~/.bashrc
batch=( 1 4 16 64 256 1024 4096 16384 )
layers=( 64 128 512 2048 8192 )
iter=( 1000 )
for b in "${batch[@]}"
do
for l in "${layers[@]}"
do
for i in "${iter[@]}"
do
../a.out $i $b $l | tee "data/cc_mkl_cpu_$b-$l-$i"
done
done
done
