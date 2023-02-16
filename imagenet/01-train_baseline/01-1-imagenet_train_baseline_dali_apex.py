'''
https://github.com/NVIDIA/DALI/blob/master/docs/examples/use_cases/pytorch/resnet50/main.py
'''
import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pdb
import sys
sys.path.append('../../')
import utils.common as myutils
from utils.options import *
from data import imagenet_dali
import models.imagenet as models

import numpy as np

def main():
    global best_prec1, args, checkpoint, logger
    # args = myutils.parser.parse_args()

    if args.evaluate:
        logger = myutils.get_logger(os.path.join(args.job_dir,'eval_log.log'))
    elif args.local_rank==0:
        checkpoint = myutils.checkpoint(args)
        logger = myutils.get_logger(os.path.join(args.job_dir,'train_log.log'))

    if not len(args.data_path):
        raise Exception("error: No data set provided")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = bool(int(os.environ['WORLD_SIZE']) > 1)

    # make apex optional
    if args.opt_level is not None or args.distributed or args.sync_bn:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    if args.local_rank==0:
        logger.info(args)
        logger.info( "opt_level = {}".format(args.opt_level) )
        logger.info( "keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32) )
        logger.info( "loss_scale = {}".format(args.loss_scale) )
        logger.info( "CUDNN VERSION: {}\n".format(torch.backends.cudnn.version()) )

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    if args.pretrained:
        if args.local_rank==0: logger.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        if args.local_rank==0: logger.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.sync_bn:
        if args.local_rank==0: logger.info("using apex synced BN")
        model = parallel.convert_syncbn_model(model)

    model = model.cuda()
    if args.local_rank==0 and not args.resume: logger.info(model)

    # Scale learning rate based on global batch size
    # args.lr = args.lr*float(args.train_batch_size*args.world_size)/256.
    if args.optimizer.lower()=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower()=='adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, 
                                     weight_decay=args.weight_decay)
    else:
        logger.info('No specific optim method')
        exit()

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.opt_level is not None:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale,min_loss_scale=1e-20
                                          )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.baseline):
                print("=> loading checkpoint '{}'".format(args.baseline))
                baseline_ckpt = torch.load(args.baseline, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = baseline_ckpt['epoch']
                best_prec1 = baseline_ckpt['best_prec1']
                try:
                    model.load_state_dict(baseline_ckpt['state_dict'])
                except:
                    logger.info("Converting checkpoint's name!!")
                    model.load_state_dict(utils.convert_keys(model, baseline_ckpt))
                optimizer.load_state_dict(baseline_ckpt['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.baseline, baseline_ckpt['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.baseline))
                exit()
        resume()

    # Data loading code
    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256
    # pdb.set_trace()
    train_loader = imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size, num_threads=4, device_id=args.local_rank, crop=224,
                                                        local_rank=args.local_rank, world_size=args.world_size, dali_cpu=args.dali_cpu)
    val_loader   = imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size, num_threads=4, device_id=args.local_rank, crop=224,
                                                        local_rank=args.local_rank, world_size=args.world_size, dali_cpu=args.dali_cpu)

    if args.evaluate:
        if os.path.isfile(args.baseline):
            print("=> loading checkpoint '{}'".format(args.baseline))
            baseline_ckpt = torch.load(args.baseline, map_location = lambda storage, loc: storage.cuda(args.gpu))
            try:
                model.load_state_dict(baseline_ckpt['state_dict'])
            except:
                logger.info("Converting checkpoint's name!!")
                model.load_state_dict(utils.convert_keys(model, baseline_ckpt))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.baseline, baseline_ckpt['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.baseline))
            exit()
        validate(val_loader, model, criterion)
        return

    total_time = myutils.AverageMeter()
    for epoch in range(args.start_epoch, args.num_epochs):
        # train for one epoch
        avg_train_time = train(train_loader, model, criterion, optimizer, epoch)
        total_time.update(avg_train_time)

        # evaluate on validation set
        [prec1, prec5] = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint.save_model({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, epoch + 1, is_best)

        train_loader.reset()
        val_loader.reset()

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = myutils.AverageMeter()
    losses = myutils.AverageMeter()
    top1 = myutils.AverageMeter()
    top5 = myutils.AverageMeter()

    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        train_loader_len = int(math.ceil(train_loader._size / args.train_batch_size))

        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        # lr = adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        lr = adjust_learning_rate(optimizer, args.lr, epoch, args.num_epochs, i, train_loader_len)

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        if args.opt_level is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
             loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = myutils.accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()

            if args.local_rank == 0:
                logger.info('Epoch: [{:>3d}][{:>4d}/{:>4d}]\t'
                      'lr {lr:.4f}\t Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Train@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Train@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, train_loader_len,
                       lr=lr, loss=losses, top1=top1, top5=top5))

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

    return batch_time.avg

def validate(val_loader, model, criterion):
    batch_time = myutils.AverageMeter()
    losses = myutils.AverageMeter()
    top1 = myutils.AverageMeter()
    top5 = myutils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        val_loader_len = int(val_loader._size / args.eval_batch_size)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = myutils.accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            logger.info('Test: [{:>4d}/{:>4d}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


# def adjust_learning_rate(optimizer, epoch, step, len_epoch):
#     """LR schedule that should yield 76% converged accuracy with batch size 256"""
#     factor = epoch // 30
#     if epoch >= 80:
#         factor = factor + 1
#     lr = args.lr*(0.1**factor)
#     """Warmup"""
#     if epoch < 5:
#         lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

def adjust_learning_rate(optimizer, init_lr, epoch, num_epochs, batch=None, nBatch=None):
    '''
    Parameters:
        batch: current batch
        nBatch: total number of batches in one epoch
    '''
    #Warmup
    if epoch < 5:
        lr = init_lr * float(1.0 + batch + epoch * nBatch) / (5. * nBatch)
    else:
        T_total = num_epochs * nBatch
        T_cur = (epoch % num_epochs) * nBatch + batch
        lr = 0.5 * init_lr * (1.0 + math.cos(math.pi * T_cur / T_total)) # cosine decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()