# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from Qtrick_architecture.clock_driven.neuron import Q_IFNode
import collections
import argparse
import datetime
import json
import numpy as np
import time
from pathlib import Path
from copy import deepcopy
import tqdm
import PIL

import torch
import torch.nn as nn
# import torchinfo
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm

# assert timm.__version__ == "0.5.4"  # version check
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
# from dvs_gait_utils.dvs_gait import create_gait_datasets


import util.lr_decay_spikformer as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_Q_scaling_dvs_for_firing_num,models_CAspikformer
# import models_sew
# import models_vs

from hardvs_dataset1 import HARDVS_dataset, EventMix
from engine_finetune_firing_num import train_one_epoch, evaluate
from timm.data import create_loader


fr_dict = {}
iter = 1350 #计算迭代轮次 #后面的数据是sample

i = 0

fr_dict_d0 = {}
fr_dict_d1 = {}
fr_dict_d2 = {}
fr_dict_d3 = {}
fr_dict_d4 = {}
fr_dict_d5 = {}
fr_dict_d6 = {}
fr_dict_d7 = {}
fr_dict_d8 = {}
def forward_hook_fn(module, input, output):  # 计算每一层的发放率
    global i
    if module.name == 'block3.2.attn.attn_spike':
        i = i+1
    x_shape = torch.tensor(list(output.shape))
    z = torch.nonzero(output)
    all_neural = torch.prod(x_shape)
    if module.name  not in fr_dict.keys():
        fr_dict[module.name] = output.detach().mean() / iter
        result0 = torch.where(output == 0,1.0, 0).mean()  # 满足条件返回a，不满足返回0
        result1 = torch.where(output==1, 1.0, 0).mean()   # 满足条件返回a，不满足返回0
        result2 = torch.where(output==2, 1.0, 0).mean()
        result3 = torch.where(output == 3, 1.0, 0).mean()
        result4 = torch.where(output == 4, 1.0, 0).mean()
        result5 = torch.where(output==5, 1.0, 0).mean()   # 满足条件返回a，不满足返回0
        result6 = torch.where(output==6, 1.0, 0).mean()
        result7 = torch.where(output == 7, 1.0, 0).mean()
        result8 = torch.where(output == 8, 1.0, 0).mean()

        fr_dict_d0[module.name] = result0 / iter
        fr_dict_d1[module.name] = result1 / iter
        fr_dict_d2[module.name] = result2 / iter
        fr_dict_d3[module.name] = result3 / iter
        fr_dict_d4[module.name] = result4 / iter
        fr_dict_d5[module.name] = result5 / iter
        fr_dict_d6[module.name] = result6 / iter
        fr_dict_d7[module.name] = result7 / iter
        fr_dict_d8[module.name] = result8 / iter
    else:
        fr_dict[module.name] = fr_dict[module.name] + output.detach().mean() / iter
        result0 = torch.where(output == 0, 1.0, 0).mean()  # 满足条件返回a，不满足返回0
        result1 = torch.where(output == 1, 1.0, 0).mean()  # 满足条件返回a，不满足返回0
        result2 = torch.where(output == 2, 1.0, 0).mean()
        result3 = torch.where(output == 3, 1.0, 0).mean()
        result4 = torch.where(output == 4, 1.0, 0).mean()

        result5 = torch.where(output==5, 1.0, 0).mean()   # 满足条件返回a，不满足返回0
        result6 = torch.where(output==6, 1.0, 0).mean()
        result7 = torch.where(output == 7, 1.0, 0).mean()
        result8 = torch.where(output == 8, 1.0, 0).mean()

        fr_dict_d0[module.name] = fr_dict_d0[module.name]+ result0 / iter
        fr_dict_d1[module.name] = fr_dict_d1[module.name]+ result1 / iter
        fr_dict_d2[module.name] = fr_dict_d2[module.name]+ result2 / iter
        fr_dict_d3[module.name] = fr_dict_d3[module.name]+ result3 / iter
        fr_dict_d4[module.name] = fr_dict_d4[module.name]+ result4 / iter
        fr_dict_d5[module.name] = fr_dict_d5[module.name]+ result5 / iter
        fr_dict_d6[module.name] = fr_dict_d6[module.name]+ result6 / iter
        fr_dict_d7[module.name] = fr_dict_d7[module.name]+ result7 / iter
        fr_dict_d8[module.name] = fr_dict_d8[module.name]+ result8 / iter

#======================


class ModelEmaV2SNN(nn.Module):
    """Model Exponential Moving Average V2 for SNN

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2SNN, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_k, ema_v, model_k, model_v in zip(
                self.module.state_dict().keys(),
                self.module.state_dict().values(),
                model.state_dict().keys(),
                model.state_dict().values(),
            ):
                if ema_k == "v" or model_k == "v":
                    continue
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
        
def get_args_parser():
    # important params
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=148,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)  # 20/30(T=4)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--data_path", default="/raid/ligq/imagenet1-k/", type=str, help="dataset path"
    )

    parser.add_argument(
        "--dataset", default="gesture", type=str, help="dataset"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="Efficient_Spiking_Transformer_scaling_8_19M",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--model_mode",
        default="ms",
        type=str,
        help="Mode of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=6e-4,
        metavar="LR",  # 1e-5,2e-5(T=4)
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1.0,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params

    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument("--time_steps", default=8, type=int)

    # Dataset parameters

    parser.add_argument(
        "--nb_classes",
        default=300,
        type=int,
        help="number of the classification types",
    )

    parser.add_argument(
        "--output_dir",
        default="/raid/ligq/htx/spikemae/output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="/raid/ligq/htx/spikemae/output_dir",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default=None, help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", default=True, help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def main(args):

    misc.init_distributed_mode(args)
    args.model = "Efficient_Spiking_Transformer_scaling_2_8M_gesture"
    # args.dataset = 'gesture'
    # args.batch_size = 16
    # args.data_path = '/raid/ligq/DVS_Gesture_dataset'
    # # args.finetune = './outputs/Efficient_Spiking_Transformer_scaling_2_8M_DVSGesture_T16x1_b16_lr3e-4_mlr1e-5_wd6e-2_LIF/checkpoint-157.pth'
    # args.finetune = './outputs/Efficient_Spiking_Transformer_scaling_2_8M_DVSGesture_T16x8_b16_lr3e-4_mlr1e-5_wd6e-2/checkpoint-80.pth'
    
    # args.model = "Efficient_Spiking_Transformer_scaling_8_19M"
    # args.dataset = 'gait-day'
    # args.batch_size = 8
    # args.time_steps = 4
    # args.finetune = './outputs/Efficient_Spiking_Transformer_scaling_2_8M_DVSGait_day_T36x8_LIF/checkpoint-123.pth'
    # args.finetune = './output_dir/hardvs_8bit/checkpoint-288.pth'
    print("ARGS.FINETUNE:",args.finetune)
    
    # args.finetune = './outputs/31M_baseline/checkpoint-best.pth'
    # args.data_path = '/storage/pony/dongzhanguo/datasets_final/imagenet'
    # args.finetune = './31M_checkpoint-best.pth'
    # args.batch_size = 30
    # args.time_steps = 1
    # args.resume = './output_dir/checkpoint-32.pth'

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transforms_train = transforms.Compose([
        transforms.Resize(
            256, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        # RandomErasing(args.reprob, mode=args.remode, max_count=args.recount, device="cpu"),
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(
            256, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset_train = HARDVS_dataset(root_path="/lxh/lxh_data/HARDVS/rawframes", train=True, img_size=224, T=8,
                                   txt_path="/lxh/lxh_data/HARDVS/list", pic_tranform=transforms_train)
    dataset_val = HARDVS_dataset(root_path="/lxh/lxh_data/HARDVS/rawframes", train=False, img_size=224, T=8,
                                 txt_path="/lxh/lxh_data/HARDVS/list", pic_tranform=transforms_test)
    # NOTE: only for HARDVS
    mix = EventMix(sensor_size=(224, 224, 2), num_classes=300, T=8)

    # dataset_train = build_dataset(is_train=True, args=args)
    # dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )
    model = models_CAspikformer.__dict__[args.model]()
    # model = models_Q_scaling_dvs_for_firing_num.__dict__[args.model]()


    checkpoint = torch.load(args.finetune, map_location="cpu")

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint["model"]

    # checkpoint_model = checkpoint["model_ema"] #11111
    # new_order = collections.OrderedDict() #用于去除掉ema前面的order
    # for key,value in checkpoint_model.items():
    #     if key[:7] == 'module.':
    #         new_order[key[7:]] = value
    # checkpoint_model = new_order

    for n, m in model.named_modules():
        if isinstance(m, Q_IFNode) :
            print(n)
            m.name = n
            m.register_forward_hook(forward_hook_fn)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)


    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))



    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
        
    model_ema = ModelEmaV2SNN(model, decay=0.999, device=None)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        # no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr) # lamb
    optimizer = optim_factory.Lamb(param_groups, trust_clip=True, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    ema_test_stats = evaluate(data_loader_val, model_ema.module, device, args.time_steps, args.eval)
    print(
        f"Accuracy of the network on the {len(dataset_val)} test images: {ema_test_stats['acc1']:.1f}%"
    )
    print("firing_rate:",fr_dict)
    print("fr_dict_d1:", fr_dict_d1)
    print("fr_dict_d2:", fr_dict_d2)
    print("fr_dict_d3:", fr_dict_d3)
    print("fr_dict_d4:", fr_dict_d4)
    print("fr_dict_d5:", fr_dict_d5)
    print("fr_dict_d6:", fr_dict_d6)
    print("fr_dict_d7:", fr_dict_d7)
    print("fr_dict_d8:", fr_dict_d8)

    # test_stats = evaluate(data_loader_val, model, device, args.time_steps, args.eval)
    # print(
    #     f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
    # )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
