import argparse
import os
import shutil
import time
import math
import pickle
import pdb
import random
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from pathlib import Path

import torchvision.transforms as transforms
import torchvision.models as models

from timm.models import create_model
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from loss import ContrastiveLoss, TripletLoss
from datasets.custom import TuplesDataset
from datasets.datahelpers import collate_tuples


def get_args_parser():
    parser = argparse.ArgumentParser('Training Vision Transformers for Image Retrieval', add_help=False)

    # Model parameters
    parser.add_argument('--task', default='category',choices=['category','purticular'])
    parser.add_argument('--model', default='deit_small_distilled_patch16_224', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME',help='name of the latest checkpoint (default: None)')
    # Optimizer parameters
    parser.add_argument('--max-iter', default=2_000, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (3e-5 for category level)')
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')

    # Dataset parameters
    parser.add_argument('--dataset', default='cub200', choices=['cub200', 'sop', 'inshop'], type=str, help='dataset path')
    parser.add_argument('--data-path', default='/data/CUB_200_2011', type=str, help='dataset path')
    parser.add_argument('--m', default=0, type=int, help="sample m images per class")
    parser.add_argument('--rank', default=[1, 2, 4, 8], nargs="+", type=int, help="compute recall@r")
    parser.add_argument('--num-workers', default=16, type=int)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Loss parameters
    parser.add_argument('--loss',default='contrastive',choices=['contrastive', 'triplet'])
    parser.add_argument('--loss-margin', '-lm', default=0.7, type=float, help='loss margin: (default: 0.7)')
    parser.add_argument('--lambda-reg', type=float, default=0.7, help="regularization strength")
    parser.add_argument('--margin', type=float, default=0.5,
                        help="negative margin of contrastive loss(beta)")

    # xbm parameters
    parser.add_argument('--memory-ratio', type=float, default=1.0, help="size of the xbm queue")
    parser.add_argument('--encoder-momentum', type=float, default=None,
                        help="momentum for the key encoder (0.999 for In-Shop dataset)")

    # MISC
    parser.add_argument('--logging-freq', type=int, default=50)
    parser.add_argument('--output-dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--log-dir', default='./logs', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # train/val options specific for image retrieval learning
    parser.add_argument('--neg-num', '-nn', default=5, type=int, metavar='N',
                        help='number of negative image per train/val tuple (default: 5)')
    parser.add_argument('--query-size', '-qs', default=2000, type=int, metavar='N',
                        help='number of queries randomly drawn per one train epoch (default: 2000)')
    parser.add_argument('--pool-size', '-ps', default=20000, type=int, metavar='N',
                        help='size of the pool for hard negative mining (default: 20000)')

    return parser

min_loss = float('inf')

def main(args):
    global min_loss

    logging.info("=" * 20 + " training arguments " + "=" * 20)
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
    logging.info("=" * 60)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

   


    # create export dir if it doesnt exist
    dataset_train, dataset_query, dataset_gallery = get_dataset(args)
    logging.info(f"Number of training examples: {len(dataset_train)}")
    logging.info(f"Number of query examples: {len(dataset_query)}")

    # set cuda visible device
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # set random seeds
    # TODO: maybe pass as argument in future implementation?
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # initialize model
     # get model
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=0,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    # if args.pretrained:
    #     print(">> Using pre-trained model '{}'".format(args.arch))
    # else:
    #     print(">> Using model from scratch (random weights) '{}'".format(args.arch))
    # model_params = {}
    # model_params['architecture'] = args.arch
    # model_params['pooling'] = args.pool
    # model_params['local_whitening'] = args.local_whitening
    # model_params['regional'] = args.regional
    # model_params['whitening'] = args.whitening
    # # model_params['mean'] = ...  # will use default
    # # model_params['std'] = ...  # will use default
    # model_params['pretrained'] = args.pretrained
    # model = init_network(model_params)

    # move network to gpu
    momentum_encoder = None
    if args.encoder_momentum is not None:
        momentum_encoder = create_model(
            args.model,
            num_classes=0,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
        for param_q, param_k in zip(model.parameters(), momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        momentum_encoder.to(device)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of params: {round(n_parameters / 1_000_000, 2):.2f} M')

    # define loss function (criterion) and optimizer
    if args.loss == 'contrastive':
        criterion = ContrastiveLoss(margin=args.loss_margin).cuda()
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.loss_margin).cuda()
    else:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))
    
    optimizer = create_optimizer(args, model)
    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    # parameters split into features, pool, whitening 
    # IMPORTANT: no weight decay for pooling parameter p in GeM or regional-GeM
    # parameters = []
    # # add feature parameters
    # parameters.append({'params': model.features.parameters()})
    # # add local whitening if exists
    # if model.lwhiten is not None:
    #     parameters.append({'params': model.lwhiten.parameters()})
    # # add pooling parameters (or regional whitening which is part of the pooling layer!)
    # if not args.regional:
    #     # global, only pooling parameter p weight decay should be 0
    #     if args.pool == 'gem':
    #         parameters.append({'params': model.pool.parameters(), 'lr': args.lr*10, 'weight_decay': 0})
    #     elif args.pool == 'gemmp':
    #         parameters.append({'params': model.pool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
    # else:
    #     # regional, pooling parameter p weight decay should be 0, 
    #     # and we want to add regional whitening if it is there
    #     if args.pool == 'gem':
    #         parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*10, 'weight_decay': 0})
    #     elif args.pool == 'gemmp':
    #         parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
    #     if model.pool.whiten is not None:
    #         parameters.append({'params': model.pool.whiten.parameters()})
    # # add final whitening if exists
    # if model.whiten is not None:
    #     parameters.append({'params': model.whiten.parameters()})

    # define optimizer
    # if args.optimizer == 'sgd':
    #     optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # elif args.optimizer == 'adam':
    #     optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    # define learning rate decay schedule
    # TODO: maybe pass as argument in future implementation?
    # exp_decay = math.exp(-0.01)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # important not to forget scheduler updating
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    # Data loading code
    # normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    train_dataset = TuplesDataset(
        data_root=args.data_path,
        mode='train',
        imsize=args.input_size,
        nnum=args.neg_num,
        qsize=args.query_size,
        poolsize=args.pool_size,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )
    if args.val:
        val_dataset = TuplesDataset(
            data_root=args.data_path,
            mode='val',
            imsize=args.input_size,
            nnum=args.neg_num,
            qsize=float('Inf'),
            poolsize=float('Inf')
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            drop_last=True, collate_fn=collate_tuples
        )

    # evaluate the network before starting
    # this might not be necessary?
    #test(args.test_datasets, model)

    for epoch in range(start_epoch, args.epochs):

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        scheduler.step()
        # # debug printing to check if everything ok
        # lr_feat = optimizer.param_groups[0]['lr']
        # lr_pool = optimizer.param_groups[1]['lr']
        # print('>> Features lr: {:.2e}; Pooling lr: {:.2e}'.format(lr_feat, lr_pool))

        # train for one epoch on train set
        loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if args.val:
            with torch.no_grad():
                loss = validate(val_loader, model, criterion, epoch)

        # evaluate on test datasets every test_freq epochs
        # if (epoch + 1) % args.test_freq == 0:
        #     with torch.no_grad():
        #         test(args.test_datasets, model)

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'meta': model.meta,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.directory)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for training
    avg_neg_distance = train_loader.dataset.create_epoch_tuples(model)

    # switch to train mode
    model.train()
    model.apply(set_batchnorm_eval)

    # zero out gradients
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple

        for q in range(nq):
            output = torch.zeros(384, ni).cuda()
            for imi in range(ni):

                # compute output vector for image imi
                output[:, imi] = model(input[q][imi].cuda()).squeeze()

            # reducing memory consumption:
            # compute loss for this query tuple only
            # then, do backward pass for one tuple only
            # each backward pass gradients will be accumulated
            # the optimization step is performed for the full batch later
            loss = criterion(output, target[q].cuda())
            losses.update(loss.item())
            loss.backward()

        if (i + 1) % args.update_every == 0:
            # do one step for multiple batches
            # accumulated gradients are used
            optimizer.step()
            # zero out gradients so we can 
            # accumulate new ones over batches
            optimizer.zero_grad()
            # print('>> Train: [{0}][{1}/{2}]\t'
            #       'Weight update performed'.format(
            #        epoch+1, i+1, len(train_loader)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(train_loader):
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    avg_neg_distance = val_loader.dataset.create_epoch_tuples(model)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        output = torch.zeros(384, nq*ni).cuda()

        for q in range(nq):
            for imi in range(ni):

                # compute output vector for image imi of query q
                output[:, q*ni + imi] = model(input[q][imi].cuda()).squeeze()

        # no need to reduce memory consumption (no backward pass):
        # compute loss for the full batch
        loss = criterion(output, torch.cat(target).cuda())

        # record loss
        losses.update(loss.item()/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(val_loader):
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg

# def test(datasets, net):

#     print('>> Evaluating network on test datasets...')

#     # for testing we use image size of max 1024
#     image_size = 1024

#     # moving network to gpu and eval mode
#     net.cuda()
#     net.eval()
#     # set up the transform
#     normalize = transforms.Normalize(
#         mean=net.meta['mean'],
#         std=net.meta['std']
#     )
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])

#     # evaluate on test datasets
#     datasets = args.test_datasets.split(',')
#     for dataset in datasets: 
#         start = time.time()

#         print('>> {}: Extracting...'.format(dataset))

#         # prepare config structure for the test dataset
#         cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
#         images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
#         qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
#         bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        
#         # extract database and query vectors
#         print('>> {}: database images...'.format(dataset))
#         vecs = extract_vectors(net, images, image_size, transform)  # implemented with torch.no_grad
#         print('>> {}: query images...'.format(dataset))
#         qvecs = extract_vectors(net, qimages, image_size, transform, bbxs)  # implemented with torch.no_grad
        
#         print('>> {}: Evaluating...'.format(dataset))

#         # convert to numpy
#         vecs = vecs.numpy()
#         qvecs = qvecs.numpy()

#         # search, rank, and print
#         scores = np.dot(vecs.T, qvecs)
#         ranks = np.argsort(-scores, axis=0)
#         compute_map_and_print(dataset, ranks, cfg['gnd'])
    
        
#         print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
            # p.requires_grad = False


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser('Training Vision Transformers for Image Retrieval', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, args.dataset)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, args.dataset)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
