# python built-in library
import os
import argparse
import time
from multiprocessing import Manager
# 3rd party library
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
# own code
from model import build_model
from dataset import KaggleDataset, Compose
from helper import config, AverageMeter, iou_mean, save_ckpt, load_ckpt
from loss import contour_criterion, focal_criterion,  mse_criterion, regularizer, focal_pixel_criterion
from valid import inference, unpack_data, get_iou

def main(resume=True, n_epoch=None, learn_rate=None):
    model_name = config['param']['model'] # helper.py line26~32 with configparser (https://docs.python.org/3/library/configparser.html) Q:meaning of ['param']['model']
    if learn_rate is None:
        learn_rate = config['param'].getfloat('learn_rate')
    width = config.getint(model_name, 'width')
    weight_map = config['param'].getboolean('weight_map')
    c = config['train']
    log_name = c.get('log_name')
    n_batch = c.getint('n_batch')
    n_worker = c.getint('n_worker')
    n_cv_epoch = c.getint('n_cv_epoch')
    if model_name == 'da_unet' or model_name == 'ynet' or model_name == 'camynet':
        domain_adaptation = True
    else:
        domain_adaptation = False
    if domain_adaptation:
        target_data = config[model_name]['target_data']
    if model_name == 'ynet' or model_name == 'camynet':
        mode = config[model_name]['mode']
        if mode == 'pretrain':
            print('pretrain mode')
        elif mode == 'train':
            print('train mode')
        elif mode == 'combine':
            print('combined training mode')
    if n_epoch is None:
        n_epoch = c.getint('n_epoch')
    balance_group = c.getboolean('balance_group')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

    model = build_model(model_name) # model.py line654
    model = model.to(device)  # class UNet(nn.Module) to()function: https://pytorch.org/docs/stable/nn.html  Q: what does to() do
    models = []
    models.append(model)
    # define optimizer
    optimizer = torch.optim.Adam(  # https://pytorch.org/docs/stable/optim.html
        filter(lambda p: p.requires_grad, model.parameters()),  # Q: what is filter() and p
        lr=args.learn_rate,
        weight_decay=1e-6
        )

    # dataloader workers are forked process thus we need a IPC manager to keep cache in same memory space
    manager = Manager()  # https://docs.python.org/3/library/multiprocessing.html#multiprocessing-managers
    cache = manager.dict()
    compose = Compose()  # dataset.py line125
    # prepare dataset
    if os.path.exists('data/valid'):
        # advance mode: use valid folder as CV
        source_dataset = KaggleDataset('data/train', 'csv_file_s', transform=compose, cache=cache)  # dataset.py line28
        if domain_adaptation:
            target_dataset = KaggleDataset('data/'+target_data, 'csv_file_t', transform=compose, cache=cache)
        else:
            valid_dataset = KaggleDataset('data/valid', 'csv_file_v', transform=compose, cache=cache)
    else:
        # auto mode: split part of train dataset as CV
        source_dataset = KaggleDataset('data/train', 'csv_file_s', transform=compose, cache=cache, use_filter=True)
        if domain_adaptation:
            source_dataset, target_dataset = source_dataset.split()
        else:
            source_dataset, valid_dataset = source_dataset.split()
    # add stage1 and stage2 testing set dataset
    resize = not config['valid'].getboolean('pred_orig_size')
    compose = Compose(augment=False, resize=resize)
    s1test_dir = os.path.join('data', 'test')
    s2test_dir = os.path.join('data', 'valid')
    if os.path.exists(s1test_dir):
        datas1test = KaggleDataset(s1test_dir, 'csv_file_s', transform=compose)
    if os.path.exists(s2test_dir):
        datas2test = KaggleDataset(s2test_dir, 'csv_file_s', transform=compose)
    # decide whether to balance training set
    if balance_group:  # Q: what is the meaning
        weights, ratio = source_dataset.class_weight() # dataset.py line116
        # Len of weights is number of original epoch samples. 
        # After oversample balance, majority class will be under-sampled (least sampled)
        # Multipling raito is to gain chance for each sample to be visited at least once in each epoch 
        sampler = WeightedRandomSampler(weights, int(len(weights) * ratio))  # https://pytorch.org/docs/stable/data.html
        if domain_adaptation:
            weights_target, ratio_target = source_dataset.class_weight()
            sampler_target = WeightedRandomSampler(weights_target, int(len(weights_target) * ratio_target))
    else:
        sampler = RandomSampler(source_dataset)
        if domain_adaptation:
            sampler_target = RandomSampler(target_dataset)
    # data loader
    source_loader = DataLoader(  # https://pytorch.org/docs/stable/data.html
        source_dataset,
        sampler=sampler,
        batch_size=n_batch,
        num_workers=n_worker,
        pin_memory=torch.cuda.is_available())
    if domain_adaptation:
        target_loader = DataLoader(
            target_dataset,
            sampler=sampler_target,
            batch_size=n_batch,
            num_workers=n_worker,
            pin_memory=torch.cuda.is_available())
    else:
        valid_loader = DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size=n_batch,
            num_workers=n_worker)

    # resume checkpoint
    start_epoch = iou_s = iou_t = iou_cv = 0
    if resume:
        start_epoch = load_ckpt(model, optimizer)  # helper.py line230
    if start_epoch == 0:
        print('Grand new training ...')

    # put model to GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  # https://pytorch.org/docs/stable/nn.html

    # decide log directory name
    log_dir = os.path.join(
        'logs', log_name, '{}-{}'.format(model_name, width),
        'ep_{},{}-lr_{}'.format(
            start_epoch,
            n_epoch + start_epoch,
            learn_rate,
        )
    )

    with SummaryWriter(log_dir) as writer:  # https://tensorboardx.readthedocs.io/en/latest/tutorial.html#create-a-summary-writer
        if start_epoch == 0 and False:
            # dump graph only for very first training, disable by default
            dump_graph(model, writer, n_batch, width)  # line116
        print('Training started...')
        for epoch in range(start_epoch + 1, n_epoch + start_epoch + 1): # 1 base
            if domain_adaptation:
                # copied from https://github.com/jvanvugt/pytorch-domain-adaptation
                batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))
                iou_s = train(batch_iterator, model, optimizer, epoch, writer, max(len(source_loader),len(target_loader)))
            else:
                iou_s = train(source_loader, model, optimizer, epoch, writer, len(source_loader))  # line146
            if domain_adaptation:
                if len(target_dataset) > 0 and epoch % n_cv_epoch == 0:
                    with torch.no_grad():  # https://pytorch.org/docs/stable/_modules/torch/autograd/grad_mode.html
                        iou_cv = valid(target_loader, model, epoch, writer, len(target_loader))
                        if os.path.exists(s1test_dir):
                            train_inference(datas1test, models, resize, compose, epoch, writer, tbprefix = 'Stage1')
                        if os.path.exists(s2test_dir):
                            train_inference(datas2test, models, resize, compose, epoch, writer, tbprefix = 'Stage2')
            else:
                if len(valid_dataset) > 0 and epoch % n_cv_epoch == 0:
                    with torch.no_grad():
                        iou_cv = valid(valid_loader, model, epoch, writer, len(source_loader))  # line220
                        if os.path.exists(s1test_dir):
                            train_inference(datas1test, models, resize, compose, epoch, writer, tbprefix = 'Stage1')
                        if os.path.exists(s2test_dir):
                            train_inference(datas2test, models, resize, compose, epoch, writer, tbprefix = 'Stage2')

            save_ckpt(model, optimizer, epoch, iou_s, iou_cv)
        print('Training finished...')

def dump_graph(model, writer, n_batch, width):
    # Prerequisite
    # $ sudo apt-get install libprotobuf-dev protobuf-compiler
    # $ pip3 install onnx
    print('Dump model graph...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.rand(n_batch, 3, width, width, device=device)
    torch.onnx.export(model, dummy_input, "checkpoint/model.pb", verbose=False)
    writer.add_graph_onnx("checkpoint/model.pb")

def train_inference(dataset, models, resize, compose, epoch ,writer, tbprefix):
    iou_test = AverageMeter()
    for data in tqdm(dataset):  # https://tqdm.github.io/docs/tqdm/
        uid, y, y_c, y_m = inference(data, models, resize)
        x, gt, gt_s, gt_c, gt_m = unpack_data(data, compose, resize)
        iou = get_iou(y, y_c, y_m, gt)
        iou_test.update(iou, 1)
    writer.add_scalar('testing/' + tbprefix + '_instance_iou', iou_test.avg, epoch)

def train(loader, model, optimizer, epoch, writer, n_step):
    batch_time = AverageMeter()  # helper.py line35
    data_time = AverageMeter()
    losses = AverageMeter()
    iou = AverageMeter()   # semantic IoU
    iou_c = AverageMeter() # contour IoU
    iou_m = AverageMeter() # marker IoU
    print_freq = config['train'].getfloat('print_freq')
    only_contour = config['contour'].getboolean('exclusive')
    weight_map = config['param'].getboolean('weight_map')
    model_name = config['param']['model']
    with_contour = config.getboolean(model_name, 'branch_contour')
    with_marker = config.getboolean(model_name, 'branch_marker')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'da_unet' or model_name == 'ynet' or model_name == 'camynet':
        domain_adaptation = True
        reg = config[model_name]['regularizer']
        if model_name == 'ynet' or model_name == 'camynet':
            mode = config[model_name]['mode']
            lamb = config[model_name]['lamb'].split(',')
    else:
        domain_adaptation = False

    # Sets the module in training mode.
    model.train()  # https://pytorch.org/docs/stable/nn.html
    end = time.time()
    for i in range(n_step):
        if domain_adaptation:
            data_s, data_t = next(loader)
        else:
            data_s = next(iter(loader))
        # measure data loading time
        data_time.update(time.time() - end)
        # split sample data
        inputs_s = data_s['image'].to(device)
        labels_s = data_s['label'].to(device)
        labels_c_s = data_s['label_c'].to(device)
        labels_m_s = data_s['label_m'].to(device)
        if domain_adaptation:
            inputs_t = data_t['image'].to(device)
        # get loss weight
        weights = None
        if weight_map and 'weight' in data_s:
            weights = data_s['weight'].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward step
        if model_name == 'da_unet':
            outputs, outputs_t, feature_map_s, feature_map_t = model([inputs_s, inputs_t], 'train')
            # add here if there is anything like 'da_camunet' or something
        elif model_name == 'ynet':
            if mode == 'pretrain':
                outputs, feature_map_s, feature_map_t = model([inputs_s, inputs_t], mode)
            elif mode == 'combine':
                outputs, rec_s, rec_t, feature_map_s, feature_map_t = model([inputs_s, inputs_t], mode)
            else:
                outputs, rec_s, rec_t = model([inputs_s, inputs_t], mode)
        elif model_name == 'camynet':
            if mode == 'pretrain':
                outputs, outputs_c, outputs_m, feature_map_s, feature_map_t = model([inputs_s, inputs_t], mode)
            elif mode == 'combine':
                outputs, outputs_c, outputs_m, rec_s, rec_t, feature_map_s, feature_map_t = model([inputs_s, inputs_t], mode)
            else:
                outputs, outputs_c, outputs_m, rec_s, rec_t = model([inputs_s, inputs_t], mode)
        else:
            outputs = model(inputs_s)
            if with_contour and with_marker:
                outputs, outputs_c, outputs_m = outputs
            elif with_contour:
                outputs, outputs_c = outputs
        # compute loss
        if only_contour:
            loss = contour_criterion(outputs, labels_c_s)  # loss.py line86
        else:
            # weight_criterion equals to segment_criterion if weights is none
            loss = focal_criterion(outputs, labels_s, weights)  # loss.py line93
            if with_contour:
                loss += focal_criterion(outputs_c, labels_c_s, weights)
            if with_marker:
                loss += focal_criterion(outputs_m, labels_m_s, weights)
            if model_name == 'da_unet':
                loss += regularizer(feature_map_s, feature_map_t, reg)
            elif model_name == 'ynet':
                if mode == 'pretrain' or mode == 'combine':
                    for j in range(len(lamb)):
                        if lamb[j] != '0':
                            loss += float(lamb[j])*regularizer(feature_map_s[j], feature_map_s[j], reg)
                if mode == 'train' or mode == 'combine':
                    loss += 0.001*(mse_criterion(rec_s, inputs_s) + mse_criterion(rec_t, inputs_t))
        # compute gradient and do backward step
        loss.backward()  # Q: cannot find backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        # NOT instance-level IoU in training phase, for better speed & instance separation handled in post-processing
        losses.update(loss.item(), inputs_s.size(0))
        if only_contour:
            batch_iou = iou_mean(outputs, labels_c_s)  # helper.py line104
        else:
            batch_iou = iou_mean(outputs, labels_s)
        iou.update(batch_iou, inputs_s.size(0))
        if with_contour:
            batch_iou_c = iou_mean(outputs_c, labels_c_s)
            iou_c.update(batch_iou_c, inputs_s.size(0))
        if with_marker:
            batch_iou_m = iou_mean(outputs_m, labels_m_s)
            iou_m.update(batch_iou_m, inputs_s.size(0))
        # log to summary
        #step = i + epoch * n_step
        #writer.add_scalar('training/loss', loss.item(), step)
        #writer.add_scalar('training/batch_elapse', batch_time.val, step)
        #writer.add_scalar('training/batch_iou', iou.val, step)
        #writer.add_scalar('training/batch_iou_c', iou_c.val, step)
        #writer.add_scalar('training/batch_iou_m', iou_m.val, step)
        if (i + 1) % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time: {batch_time.avg:.2f} (io: {data_time.avg:.2f})\t'
                'Loss: {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                'IoU: {iou.avg:.3f} (Coutour: {iou_c.avg:.3f}, Marker: {iou_m.avg:.3f})\t'
                .format(
                    epoch, i, n_step, batch_time=batch_time,
                    data_time=data_time, loss=losses, iou=iou, iou_c=iou_c, iou_m=iou_m
                )
            )
    # end of loop, dump epoch summary
    writer.add_scalar('training/epoch_loss', losses.avg, epoch)
    writer.add_scalar('training/epoch_iou', iou.avg, epoch)
    writer.add_scalar('training/epoch_iou_c', iou_c.avg, epoch)
    writer.add_scalar('training/epoch_iou_m', iou_m.avg, epoch)
    return iou.avg # return epoch average iou

def valid(loader, model, epoch, writer, n_step):
    iou = AverageMeter()   # semantic IoU
    iou_c = AverageMeter() # contour IoU
    iou_m = AverageMeter() # marker IoU
    losses = AverageMeter()
    only_contour = config['contour'].getboolean('exclusive')
    weight_map = config['param'].getboolean('weight_map')
    model_name = config['param']['model']
    with_contour = config.getboolean(model_name, 'branch_contour')
    with_marker = config.getboolean(model_name, 'branch_marker')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'da_unet' or model_name == 'ynet' or model_name == 'camynet':
        domain_adaptation = True
    else:
        domain_adaptation = False

    # Sets the model in evaluation mode.
    model.eval()  # https://pytorch.org/docs/stable/nn.html
    for i, data in enumerate(loader):
        # get the inputs
        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        labels_c = data['label_c'].to(device)
        labels_m = data['label_m'].to(device)
        # get loss weight
        weights = None
        if weight_map and 'weight' in data:
            weights = data['weight'].to(device)
        # forward step
        if domain_adaptation:
            outputs = model(inputs, 'valid')
        else:
            outputs = model(inputs)
        if with_contour and with_marker:
            outputs, outputs_c, outputs_m = outputs
        elif with_contour:
            outputs, outputs_c = outputs
        # compute loss
        if only_contour:
            loss = contour_criterion(outputs, labels_c)
        else:
            # weight_criterion equals to segment_criterion if weights is none
            loss = focal_criterion(outputs, labels, weights)
            if with_contour:
                loss += focal_criterion(outputs_c, labels_c, weights)
            if with_marker:
                loss += focal_criterion(outputs_m, labels_m, weights)
        # measure accuracy and record loss (Non-instance level IoU)
        losses.update(loss.item(), inputs.size(0))
        if only_contour:
            batch_iou = iou_mean(outputs, labels_c)
        else:
            batch_iou = iou_mean(outputs, labels)
        iou.update(batch_iou, inputs.size(0))
        if with_contour:
            batch_iou_c = iou_mean(outputs_c, labels_c)
            iou_c.update(batch_iou_c, inputs.size(0))
        if with_marker:
            batch_iou_m = iou_mean(outputs_m, labels_m)
            iou_m.update(batch_iou_m, inputs.size(0))
    # end of loop, dump epoch summary
    writer.add_scalar('CV/epoch_loss', losses.avg, epoch)
    writer.add_scalar('CV/epoch_iou', iou.avg, epoch)
    writer.add_scalar('CV/epoch_iou_c', iou_c.avg, epoch)
    writer.add_scalar('CV/epoch_iou_m', iou_m.avg, epoch)
    print(
        'Epoch: [{0}]\t\tcross-validation\t'
        'Loss: N/A    (avg: {loss.avg:.4f})\t'
        'IoU: {iou.avg:.3f} (Coutour: {iou_c.avg:.3f}, Marker: {iou_m.avg:.3f})\t'
        .format(
            epoch, loss=losses, iou=iou, iou_c=iou_c, iou_m=iou_m
        )
    )
    return iou.avg # return epoch average iou

def loop_iterable(iterable):  # copied from https://github.com/jvanvugt/pytorch-domain-adaptation
    while True:
        yield from iterable

if __name__ == '__main__':
    learn_rate = config['param'].getfloat('learn_rate')
    n_epoch = config['train'].getint('n_epoch')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--epoch', type=int, help='run number of epoch')
    parser.add_argument('--lr', type=float, dest='learn_rate', help='learning rate')
    parser.set_defaults(resume=True, epoch=n_epoch, learn_rate=learn_rate)
    args = parser.parse_args()

    main(args.resume, args.epoch, args.learn_rate)
