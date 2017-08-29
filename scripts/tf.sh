#!/bin/bash


export CUDA_VISIBLE_DEVICES=""
source ~/.bashrc
batch=( 1 4 16 64 256 1024 4096 16384 )
layers=( 64 128 512 2048 8192 )
feed=( 0 1 )
iter=( 1000 )
for b in "${batch[@]}"
do
for l in "${layers[@]}"
do
for f in "${feed[@]}"
do
for i in "${iter[@]}"
do
/opt/intel/intelpython2/bin/python ../tf/simple_cpu_nn.py --batch_size=$b --layers=$l,512,128,1 --output="data/tf_cpu_$b-$l-$f-$i" --num_layers=4 --feed=$f --num_iter=$i
done
done
done
done
