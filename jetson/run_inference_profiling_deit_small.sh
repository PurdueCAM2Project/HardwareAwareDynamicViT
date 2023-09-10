#!/bin/bash
python infer.py --eval-tensorboard --forward-pass-count 200 --model deit-s --batch_size 1 --data_path /usr/local/data/ImageNet1K/ILSVRC/Data/CLS-LOC/ --num_workers 4 --base_rate 0.7 --model_path bin/dynamic-vit_384_r0.7.pth 
