# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from spikingjelly.clock_driven import functional


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        functional.reset_net(model)
        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def calc_non_zero_rate(s_dict, nz_dict, idx, t):
    for k, v_ in s_dict.items():
        v = v_[t, ...]
        x_shape = torch.tensor(list(v.shape))
        all_neural = torch.prod(x_shape)
        z = torch.nonzero(v)
        if k in nz_dict.keys():
            nz_dict[k] += (z.shape[0] / all_neural).item() / idx
        else:
            nz_dict[k] = (z.shape[0] / all_neural).item() / idx
    return nz_dict

def calc_firing_rate(s_dict, fr_dict, idx, t):
    for k, v_ in s_dict.items():
        v = v_[t, ...]

        v = v
        if v.max() <=1 and k !='Efficient_SpikeFormer_scaling_input_spike' : #做好归一化
            v = v * 4
        # print("vmax:", v.max())

        if k in fr_dict.keys():
            fr_dict[k] += v.mean().item() / idx
        else:
            fr_dict[k] = v.mean().item() / idx
    return fr_dict


@torch.no_grad()
# def evaluate(data_loader, model, device): # 以防万一留一个备份
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = misc.MetricLogger(delimiter="  ")
#     header = "Test:"

#     # switch to evaluation mode
#     model.eval()

#     for batch in metric_logger.log_every(data_loader, 500, header):
#         images = batch[0]
#         target = batch[-1]
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         with torch.cuda.amp.autocast():
#             output = model(images)
#             loss = criterion(output, target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         functional.reset_net(model)

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
#         metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print(
#         "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
#             top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
#         )
#     )

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, time_steps=1, eval=False):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    last_idx = len(data_loader) - 1
    fr_dict = {f"t{i}": dict() for i in range(time_steps)}
    nz_dict = {f"t{i}": dict() for i in range(time_steps)}
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        last_batch = batch == last_idx

        images = batch[0]
        target = batch[-1]
        # if len(images.shape) == 6:
        #     images = images.flatten(0,1) # 去掉gait里的clip
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output, firing_dict = model(images,hook = {})
            output= model(images)

            loss = criterion(output, target)

        # if eval:
        #     for t in range(time_steps):
        #         fr_single_dict = calc_firing_rate(
        #             firing_dict, fr_dict["t" + str(t)], last_idx, t
        #         )
        #         fr_dict["t" + str(t)] = fr_single_dict
        #
        #         nz_single_dict = calc_non_zero_rate(
        #             firing_dict, nz_dict["t" + str(t)], last_idx, t
        #         )
        #         nz_dict["t" + str(t)] = nz_single_dict
                
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        functional.reset_net(model)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    
    file_path = './output_dir/19M_HARDVS_8bit_int_firing_rate.txt'
    all_firing_num = 0
    times = 0.0001

    with open(file_path, 'w') as file:
        for main_key, sub_dict in fr_dict.items():
            file.write(f'{main_key}:\n')
            for key, value in sub_dict.items():
                all_firing_num = all_firing_num + value
                times = times + 1
                file.write(f'    {key}: {value}\n')
            file.write('\n')
        all_firing_rate = all_firing_num/times
        print("path:",file_path)
        print("all_firing_rate:",all_firing_rate)
        file.write(str(all_firing_rate))

    
    
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
