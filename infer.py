# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
from functools import partial

from models.dyvit import VisionTransformerDiffPruning
from models.dylvvit import LVViTDiffPruning
from models.dyconvnext import AdaConvNeXt
from models.dyswin import AdaSwinTransformer
import utils
from typing import Any, List, Dict, Tuple

import pandas

import torch.utils.benchmark as bench

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', default='deit_small', type=str, help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default='', help='resume from checkpoint')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--base_rate', type=float, default=0.7)
    parser.add_argument('--no-progress-bar', action='store_true')
    parser.add_argument("--device", type=str, default="cuda:0")

    ###
    ### For Profiling
    ###
    parser.add_argument('--pruning-loc-mask', nargs='+', default=None)
    parser.add_argument("--output-filename-suffix", type=str, default="")
    parser.add_argument("--evaluate-only-no-accuracy", action="store_true")

    return parser

### Utility function for pruning location overriding
def list_to_int_list( list : List) -> List[int]:
    return [int(k) for k in list]

def main(args):
    ### Set matmul precision
    torch.set_float32_matmul_precision('high')

    #cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    base_rate = args.base_rate
    KEEP_RATE1 = [base_rate, base_rate ** 2, base_rate ** 3]
    KEEP_RATE2 = [base_rate, base_rate - 0.2, base_rate - 0.4]

    print(f"Creating model: {args.model}")

    ###
    ### Initialize empty pruning location mask
    ### 
    PRUNING_LOC_MASK = None

    if args.model == 'deit-s':
        PRUNING_LOC = [3,6,9]
        
        ### Check if we want to override pruning loc
        if args.pruning_loc_mask is not None:
            PRUNING_LOC_MASK = list_to_int_list(args.pruning_loc_mask)
            print('infer.py: WARNING, masking all pruning locations except {}'.format(PRUNING_LOC_MASK))

        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1, pruning_loc_mask=PRUNING_LOC_MASK
            )
    elif args.model == 'deit-256':
        PRUNING_LOC = [3,6,9] 
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
            )
    elif args.model == 'deit-b':
        PRUNING_LOC = [3,6,9] 

        ### Check if we want to override pruning loc
        if args.pruning_loc_mask is not None:
            PRUNING_LOC_MASK = list_to_int_list(args.pruning_loc_mask)
            print('infer.py: WARNING, masking all pruning locations except {}'.format(PRUNING_LOC_MASK))

        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1, pruning_loc_mask=PRUNING_LOC_MASK
            )
    elif args.model == 'lvvit-s':
        PRUNING_LOC = [4,8,12] 
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
        )
    elif args.model == 'lvvit-m':
        PRUNING_LOC = [5,10,15] 
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
        )
    elif args.model == 'convnext-t':
        PRUNING_LOC = [1,2,3]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC
        )
    elif args.model == 'convnext-s':
        PRUNING_LOC = [3,6,9]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC, 
            depths=[3, 3, 27, 3]
        )
    elif args.model == 'convnext-b':
        PRUNING_LOC = [3,6,9]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC, 
            depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
        )
    elif args.model == 'swin-t':
        PRUNING_LOC = [1,2,3]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_rate=0.0,
            pruning_loc=[1,2,3], sparse_ratio=KEEP_RATE2
        )
    elif args.model == 'swin-s':
        PRUNING_LOC = [2,4,6]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[2,4,6], sparse_ratio=KEEP_RATE2
        )
    elif args.model == 'swin-b':
        PRUNING_LOC = [2,4,6]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[2,4,6], sparse_ratio=KEEP_RATE2
        )
    else:
        raise NotImplementedError

    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    print('## model has been successfully loaded')

    model = model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    validate(args, data_loader_val, model, criterion)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

###
### Benchmark function for profiling wrapped models
### 
def benchmark_milliseconds_wrapped(
    args : argparse.Namespace, 
    x : torch.Tensor, 
    model : torch.nn.Module
    ) -> bench.Measurement:
    ### Set a minimum runtime 
    MIN_RUNTIME = 32.0

    t0 = bench.Timer(
        stmt=f"model(x)",
        globals={
            "x" : x,
            "model" : model,
        }
    )

    return t0.blocked_autorange(min_run_time=MIN_RUNTIME)

def validate(args, val_loader, model, criterion):
    ### Port to device
    model.eval().to(args.device)
    
    ### Latency measurment data
    latency_measurement = None

    ### Accuracy computation
    running_accuracy = 0.0

    ### Use TQDM instead
    dataloader_object = tqdm(val_loader)

    with torch.no_grad():
        for batch_index, (images, target) in enumerate(dataloader_object):
            ### To Device
            images = images.to(args.device)
            target = target.to(args.device)

            ### Benchmark timing 
            if batch_index == 0:
                dataloader_object.set_description(desc="Benchmarking (potentially wrapped) model...")
                latency_measurement = benchmark_milliseconds_wrapped(args, images, model)

                if args.evaluate_only_no_accuracy:
                    break

            # compute output
            output = model(images)

            ### Append running accuracy
            running_accuracy += (
                output == target
            ).sum().item() / target.shape[0]

            dataloader_object.set_description(desc=f"Recording Accuracy...", refresh=False)

    accuracy = 100.0 * running_accuracy / len(dataloader_object) if not args.evaluate_only_no_accuracy else -1.0

    latency_mean = latency_measurement.mean * 1e3
    latency_median = latency_measurement.median * 1e3
    latency_iqr = latency_measurement.iqr * 1e3

    ### Print Accuracy
    print("gpu_tail_measure.py: Accuracy is {:.3f}".format(accuracy))
    
    ### Print Latency Data
    print("gpu_tail_measure.py: Latency (ms) stats Mean/Median/IQR are {:.3f} / {:.3f} / {:.3f}".format(latency_mean, latency_median, latency_iqr))

    ### Save as .CSV data
    eval_dataframe = pandas.DataFrame(
        data={
            "Accuracy": [accuracy],
            "Avg. Latency (ms)": [latency_mean],
            "Median Latency (ms)": [latency_median],
            "Latency IQR (ms)": [latency_iqr],

        }
    ).to_csv(
        f"bin/dynamic_vit_r{args.base_rate}_{args.output_filename_suffix}_inference_data.csv",
        float_format="{:.2f}".format,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)