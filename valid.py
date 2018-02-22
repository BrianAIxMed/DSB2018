# python built-in library
import os
import argparse
import time
import csv
from multiprocessing import Manager
# 3rd party library
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage.transform import resize
from skimage.morphology import label, remove_small_objects
from tqdm import tqdm
# own code
from model import UNet, UNetVgg16, DCAN, CAUNet
from dataset import KaggleDataset, NuclearDataset, Compose
from helper import config, load_ckpt, prob_to_rles, seg_ws, iou_metric

def main(tocsv=False, save=False, valid_train=False):
    model_name = config['param']['model']
    cell_level = config['param'].getboolean('cell_level')

    if model_name == 'unet_vgg16':
        model = UNetVgg16(3, 1, fixed_vgg=True)
    elif model_name == 'dcan':
        model = DCAN(3, 1)
    elif model_name == 'caunet':
        model = CAUNet()
    else:
        model = UNet()

    if torch.cuda.is_available():
        model = model.cuda()
        # model = torch.nn.DataParallel(model).cuda()

    # Sets the model in evaluation mode.
    model.eval()

    epoch = load_ckpt(model)
    if epoch == 0:
        print("Aborted: checkpoint not found!")
        return

    # prepare dataset
    compose = Compose(augment=False)
    data_dir = 'data/stage1_train' if valid_train else 'data/stage1_test'
    if cell_level:
        dataset = NuclearDataset(data_dir, transform=compose)
    else:
        dataset = KaggleDataset(data_dir, transform=compose)
        # dataset = KaggleDataset(data_dir, transform=compose, category='Histology')
    iter = predict(model, dataset, compose)

    if tocsv:
        with open('result.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ImageId', 'EncodedPixels'])
            for uid, _, y, _, _, _, _, _ in iter:
                for rle in prob_to_rles(y):
                    writer.writerow([uid, ' '.join([str(i) for i in rle])])
    else:
        for uid, x, y, gt, y_s, gt_s, y_c, gt_c in tqdm(iter):
            if valid_train:
                show_groundtruth(uid, x, y, gt, y_s, gt_s, y_c, gt_c, save)
            else:
                show(uid, x, y, save)

def predict(model, dataset, compose, regrowth=True):
    ''' iterate dataset and yield ndarray result tuple per sample '''
    if isinstance(model, DCAN) or isinstance(model, CAUNet):
        model_name = config['param']['model']
        threshold_sgmt = config[model_name].getfloat('threshold_sgmt')
        threshold_edge = config[model_name].getfloat('threshold_edge')

    for data in dataset:
        # get prediction
        uid = data['uid']
        inputs = x = data['image']
        gt_s, gt_c, gt = data['label'], data['label_e'], data['label_gt']
        inputs = inputs.unsqueeze(0)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        if isinstance(model, DCAN) or isinstance(model, CAUNet):
            outputs_s, outputs_c = model(inputs)
            cond1 = (outputs_s >= threshold_sgmt)
            cond2 = (outputs_c < threshold_edge)
            outputs = (cond1 * cond2)
        else:
            outputs = model(inputs)

        # convert image to numpy array
        x = compose.denorm(x)
        x = compose.pil(x)
        gt_s = compose.pil(gt_s)
        gt_c = compose.pil(gt_c)
        gt = compose.pil(gt)
        if regrowth:
            x = x.resize(data['size'])
            gt_s = gt_s.resize(data['size'])
            gt_c = gt_c.resize(data['size'])
            gt = gt.resize(data['size'])

        x = np.asarray(x)
        gt_s = np.asarray(gt_s)
        gt_c = np.asarray(gt_c)
        gt = np.asarray(gt)

        # convert predict to numpy array
        if torch.cuda.is_available():
            outputs = outputs.cpu()
            if isinstance(model, DCAN) or isinstance(model, CAUNet):
                outputs_s = outputs_s.cpu()
                outputs_c = outputs_c.cpu()

        y = outputs.data.numpy()[0]
        y = np.transpose(y, (1, 2, 0))
        y = np.squeeze(y)
        if regrowth:
            y = resize(y, data['size'][::-1], mode='constant', preserve_range=True)

        if isinstance(model, DCAN) or isinstance(model, CAUNet):
            y_s = outputs_s.data.numpy()[0]
            y_s = np.transpose(y_s, (1, 2, 0))
            y_s = np.squeeze(y_s)
            y_c = outputs_c.data.numpy()[0]
            y_c = np.transpose(y_c, (1, 2, 0))
            y_c = np.squeeze(y_c)
            if regrowth:
                y_s = resize(y_s, data['size'][::-1], mode='constant', preserve_range=True)
                y_c = resize(y_c, data['size'][::-1], mode='constant', preserve_range=True)
        else:
            y_s = y_c = None

        # yield result
        yield uid, x, y, gt, y_s, gt_s, y_c, gt_c

def _make_overlay(img_array):
    img_array = img_array.astype(float)
    img_array[img_array == 0] = np.nan # workaround: matplotlib cmap mistreat vmin(1) as background(0) sometimes
    cmap = plt.get_cmap('prism') # prism for high frequence color bands
    cmap.set_bad('w', alpha=0) # map background(0) as transparent/white
    return img_array, cmap

def show(uid, x, y, save=False):
    threshold = config['param'].getfloat('threshold')
    segmentation = config['post'].getboolean('segmentation')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(14, 6))
    fig.suptitle(uid, y=1)
    ax1.set_title('Image')
    ax2.set_title('Predict, P > {}'.format(threshold))
    ax3.set_title('Region, P > {}'.format(threshold))
    ax4.set_title('Overlay, P > {}'.format(threshold))
    ax1.imshow(x, aspect='auto')
    y = y > threshold
    ax2.imshow(y, cmap='gray', aspect='auto')
    y = label(y)
    if segmentation:
        y = seg_ws(y)
    y, cmap = _make_overlay(y)
    ax3.imshow(y, cmap=cmap, aspect='auto')
    # alpha
    ax4.imshow(x, aspect='auto')
    ax4.imshow(y, cmap=cmap, alpha=0.3, aspect='auto')
    plt.tight_layout()
    if save:
        dir = predict_save_folder()
        fp = os.path.join(dir, uid + '.png')
        plt.savefig(fp)
    else:
        plt.show()

def show_groundtruth(uid, x, y, gt, y_s, gt_s, y_c, gt_c, save=False):
    threshold = config['param'].getfloat('threshold')
    segmentation = config['post'].getboolean('segmentation')
    remove_objects = config['post'].getboolean('remove_objects')
    min_object_size = config['post'].getint('min_object_size')
    model_name = config['param']['model']
    if model_name == 'dcan' or model_name == 'caunet':
        threshold_sgmt = config[model_name].getfloat('threshold_sgmt')
        threshold_edge = config[model_name].getfloat('threshold_edge')

    if y_s is not None and y_c is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 4, sharey=True, figsize=(21, 9))
    else:
        fig, ax1 = plt.subplots(1, 4, sharey=True, figsize=(14, 6))
    fig.suptitle(uid, y=1)

    ax1[0].set_title('Image')
    ax1[0].imshow(x, aspect='auto')
    y = y > threshold
    _, count = label(y, return_num=True)
    ax1[1].set_title('Final Pred, Pre#={}'.format(count))
    ax1[1].imshow(y, cmap='gray', aspect='auto')
    # overlay contour to semantic ground truth (another visualized view for instance ground truth, eg. gt)
    count = len(np.unique(gt)) - 1 # remove background
    ax1[2].set_title('Instance Lbls, #={}'.format(count))
    ax1[2].imshow(gt_s, cmap='gray', aspect='auto')
    gt_c2, cmap = _make_overlay(gt_c)
    ax1[2].imshow(gt_c2, cmap=cmap, alpha=0.7, aspect='auto')
    # overlay (applied post-processing)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    y = label(y)
    if segmentation:
        y = seg_ws(y)
    iou = iou_metric(y, gt, instance_level=True)
    _, count = label(y, return_num=True)
    ax1[3].set_title('Overlay, Post#={}, IoU={:.3f}'.format(count, iou))
    ax1[3].imshow(gt_s, cmap='gray', aspect='auto')
    y, cmap = _make_overlay(y)
    ax1[3].imshow(y, cmap=cmap, alpha=0.3, aspect='auto')

    if y_s is not None and y_c is not None:
        ax2[0].set_title('Image')
        ax2[0].imshow(x, aspect='auto')
        y_s = y_s > threshold_sgmt
        _, count = label(y_s, return_num=True)
        ax2[1].set_title('Semantic Predict, #={}'.format(count))
        ax2[1].imshow(y_s, cmap='gray', aspect='auto')
        _, count = label(gt_s, return_num=True)
        ax2[2].set_title('Semantic Lbls, #={}'.format(count))
        ax2[2].imshow(gt_s, cmap='gray', aspect='auto')
        # overlay
        iou = iou_metric(y_s, gt_s)
        ax2[3].set_title('Overlay(Semantic), IoU={:.3f}'.format(iou))
        ax2[3].imshow(gt_s, cmap='gray', aspect='auto')
        y_s = label(y_s)
        y_s, cmap = _make_overlay(y_s)
        ax2[3].imshow(y_s, cmap=cmap, alpha=0.3, aspect='auto')

        ax3[0].set_title('Image')
        ax3[0].imshow(x, aspect='auto')
        y_c = y_c > threshold_edge
        _, count = label(y_c, return_num=True)
        ax3[1].set_title('Contour Predict, #={}'.format(count))
        ax3[1].imshow(y_c, cmap='gray', aspect='auto')
        _, count = label(gt_c, return_num=True)
        ax3[2].set_title('Contour Lbls, #={}'.format(count))
        ax3[2].imshow(gt_c, cmap='gray', aspect='auto')
        # overlay
        iou = iou_metric(y_c, gt_c)
        ax3[3].set_title('Overlay(Contour), IoU={:.3f}'.format(iou))
        ax3[3].imshow(gt_c, cmap='gray', aspect='auto')
        y_c = label(y_c)
        y_c, cmap = _make_overlay(y_c)
        ax3[3].imshow(y_c, cmap=cmap, alpha=0.3, aspect='auto')

    plt.tight_layout()

    if save:
        dir = predict_save_folder()
        fp = os.path.join(dir, uid + '.png')
        plt.savefig(fp)
    else:
        plt.show()

def predict_save_folder():
    return os.path.join('data', 'predict')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=['train', 'test'], help='dataset to eval')
    parser.add_argument('--csv', dest='csv', action='store_true')
    parser.add_argument('--show', dest='csv', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.set_defaults(csv=False, save=False, dataset='test')
    args = parser.parse_args()

    if not args.csv:
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            print(err)
            print("[ERROR] No GUI library for rendering, consider to save as RLE '--csv'")
            exit(-1)

        if args.save:
            print("[INFO] Save side-by-side prediction figure in 'data/predict' folder...")
            dir = predict_save_folder()
            if not os.path.exists(dir):
                os.makedirs(dir)

    main(args.csv, args.save, args.dataset == 'train')
