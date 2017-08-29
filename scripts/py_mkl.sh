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
/home/cc/intel/intelpython2/bin/python ../pipeline.py --batch_size=$b --layers=$l,512,128,1 --output="data/py_mkl_cpu_$b-$l-$f-$i" --num_layers=4 --num_iter=$i
done
done
done
