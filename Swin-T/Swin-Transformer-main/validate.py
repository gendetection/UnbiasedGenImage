# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import csv

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

from tqdm import tqdm
from logger import create_logger


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', default = True, help='Perform evaluation only, here True since validation script')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    ## Deepfake detection specific arguments
    parser.add_argument('--csv_data_path', type=str, help='The path to the CSV file containing all the metadatqaa about the GenImage dataset')
    parser.add_argument('--base_path', type = str, help = 'path where the GenImage directoty is stored')
    parser.add_argument('--dataset', type=str,
                    help='the dataset defines which dataset is used in the dataloader -> One of classic, jpeg96, controlled or define new')
    parser.add_argument('--generator', type=str, choices=['Midjourney', 'stable_diffusion_v_1_5', 'stable_diffusion_v_1_4', 'wukong', 'ADM', 'VQDM', 'glide', 'BigGAN'],
                    help='this is the generator on which the model is trained, so it defines the genimage subset')
    # If dataset == "controlled":
    parser.add_argument('--min_size', type=int, default=None,
                    help='Only nature images in intervall [min_size, max_size] are included')
    parser.add_argument('--max_size', type=int, default=None,
                    help='Only nature images in intervall [min_size, max_size] are included')
    parser.add_argument('--jpeg_qf', type=int, default=None,
                    help='if set, all images are jpeg compressed with this quality factor')
    parser.add_argument('--class-map', default='../../class_map.txt', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file')
    parser.add_argument('--balance_train_classes', action='store_true', default=False,
                    help='whether or not to balance train data so that the class distribution is equal in ai images and nature images \
                        (number of instances per imagenet class is same in ai images and nature images)')
    parser.add_argument('--balance_val_classes', action='store_true', default=False,
                    help='whether or not to balance val data so that the class distribution is equal in ai images and nature images \
                        (number of instances per imagenet class is same in ai images and nature images)')
    parser.add_argument('--sample_qf_ai', action='store_true', default=False,
                    help='If this is set and jpeg_qf is None, the ai qf is sampled from the distribution of the qf from all natural train images')
    parser.add_argument('--resize', type=int, default=None,
                        help='if set, all images are first resized to this')
    parser.add_argument('--cropsize', type=int, default=None,
                        help='if set, all images are cropped to this size arfter resizing')
    parser.add_argument('--cropmethod', type=str, choices="['center', 'random']", default='center')
    parser.add_argument('--compress_natural', action='store_true', default=False,
                        help=' Whether to also compress the natural images with the given jpeg qf')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1 = accuracy(output, target, topk=(1,))[0]

        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    return acc1_meter.avg, loss_meter.avg


def main(config):
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    data_loader = build_loader(config)
    model = build_model(config)
    model_without_ddp = model

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        model.cuda()
        acc1, loss = validate(config, data_loader, model)
        logger.info(f"Accuracy of the network on the {len(dataset)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        model.cuda()
        acc1, loss = validate(config, data_loader, model)
        logger.info(f"Accuracy of the network on the test images: {acc1:.1f}%")

    else:
        raise ValueError("No path to pretrained model given, please consider passing such a path as an argument (--resume or --pretrained)")
      
    output_file = os.path.join(config.OUTPUT, config.TAG + "_" + config.generator + ".csv")
    with open(output_file, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the accuracy data
        csv_writer.writerow([acc1, loss])



if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)