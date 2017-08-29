#!/bin/bash


export CUDA_VISIBLE_DEVICES=""
source ~/.bashrc
batch=( 1 16 512 2048 4096 8192 )
layers=( 784 )
iter=( 1000 )
for b in "${batch[@]}"
do
for l in "${layers[@]}"
do
for i in "${iter[@]}"
do
/opt/intel/intelpython2/bin/python ../pipeline.py --batch_size=$b --layers=$l,512,128,1 --output="data/py_cpu_$b-$l-$f-$i" --num_layers=4 --num_iter=$i
done
done
done
