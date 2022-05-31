#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_GraFormer.py \
    --mesh
    --dataset_path