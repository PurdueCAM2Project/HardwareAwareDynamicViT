#!/bin/bash
python infer.py --model deit-b --batch_size 1 --data_path /usr/local/data/ImageNet1K/ILSVRC/Data/CLS-LOC/ --num_workers 4 --base_rate 0.7 --model_path bin/dyvit-b_r0.7.pth
