"""
Train a SegNet model


Usage:
python train.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                --train_path ImageSets/Segmentation/train.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir /home/SharedData/intern_sayan/PascalVOC2012/ \
                --checkpoint /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                --gpu 1

new lf usage

python ./src/train.py --data_root /home/cxt/Documents/research/lf_dope/lf_trans_dataset/synthetic/ --save_dir /home/cxt/Documents/research/lf_dope/pytorch-segnet/checkpoint/ 
                

"""

from __future__ import print_function
import argparse
from dataset import LFDataset, NUM_CLASSES
from model import SegNet
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.nn as nn

# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES + 1  # boundary around object
NUM_KEYPOINTS = 8 + 1  # 8 corners + 1 center or 2 major points + 1 center

NUM_EPOCHS = 100

LEARNING_RATE = 5e-4
BATCH_SIZE = 1


# Arguments
parser = argparse.ArgumentParser(description='Train a SegNet model')

parser.add_argument('--data_root', required=True)
# parser.add_argument('--train_path', required=True)
# parser.add_argument('--img_dir', required=True)
# parser.add_argument('--mask_dir', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--checkpoint')
parser.add_argument('--partial_preload')
parser.add_argument('--edgemap')
# parser.add_argument('--gpu', type=int)

args = parser.parse_args()



def train():
    is_better = True
    prev_loss = float('inf')

    model.train()
    print ("Batch Number:" + str(len(train_dataloader)))
    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        loss_seg_f = 0
        loss_reg_f = 0
        t_start = time.time()
        batch_id = 0
        for batch in train_dataloader:
            input_tensor = torch.autograd.Variable(batch['image'])
            seg_target_tensor = torch.autograd.Variable(batch['mask'])
            key_target_tensor = torch.autograd.Variable(batch['gt_offset'])

            if CUDA:
                input_tensor = input_tensor.cuda()
                seg_target_tensor = seg_target_tensor.cuda()
                key_target_tensor = key_target_tensor.cuda()

            seg_tensor, key_tensor = model(input_tensor)

            optimizer.zero_grad()
            loss_seg = criterion(seg_tensor, seg_target_tensor)
            loss_key = calculate_keyloss(key_tensor, key_target_tensor, seg_target_tensor)
            loss = loss_seg + loss_key

            loss.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                print("Epoch #{}\tBatch #{}\tLoss: {:.8f}\tLoss_seg: {:.8f}\tLoss_reg: {:.8f}".format(epoch + 1, batch_id, loss, loss_seg, loss_key))
            loss_f += loss.float()
            loss_seg_f += loss_seg.float()
            loss_reg_f += loss_key.float()
            prediction_f = seg_tensor.float()
            batch_id = batch_id + 1

        loss_f = loss_f / len(train_dataloader)
        loss_seg_f = loss_seg_f / len(train_dataloader)
        loss_reg_f = loss_reg_f / len(train_dataloader)
        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pth"))
            print("saved new best model")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_" + str(epoch) + ".pth"))
            print("saved new best model")
        print("Epoch #{}\tLoss: {:.8f}\tLoss_seg: {:.8f}\tLoss_reg: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, loss_seg_f, loss_reg_f, delta))


def calculate_keyloss(key_tensor, key_target_tensor, seg_target_tensor):
    # position loss: sum(seg)sum(keypoints)|delta(pos)|
    # confidence loss: sum(seg)sum(keypoints)|conf - exp(-tau * delta(pos)|

    # key_tensor: (nBatch, 3*nKeypoints (x1, x2, xN, y1, y2, yN, conf1, conf2, confN), nWidth, nHeight)
    # key_target_tensor: (nBatch, 2*nKeypoints (x1, x2, xN, y1, y2, yN), nWidth, nHeight)
    beta = 0.8
    gamma = 1 - beta
    tau = 1
    norm_factor = 10

    nBatch, nWidth, nHeight = seg_target_tensor.size()
    seg_target_tensor = seg_target_tensor.view(nBatch * nWidth * nHeight)

    nKeypoints = key_tensor.size(1) / 3
    key_tensor = key_tensor.transpose(dim0=0, dim1=1).contiguous().view(3 * nKeypoints, nBatch * nWidth * nHeight)
    key_target_tensor = key_target_tensor.transpose(dim0=0, dim1=1).contiguous().view(2 * nKeypoints, nBatch * nWidth * nHeight)

    roi_ind = seg_target_tensor.nonzero().squeeze()

    if roi_ind.size(0) == 0:  # no segmentation estimation output
        return torch.tensor(0.0).cuda()

    key_tensor = key_tensor.index_select(dim=1, index=roi_ind)
    xy_gt = key_target_tensor.index_select(dim=1, index=roi_ind)

    # key_tensor: (3*nKeypoints, sum_batch(nSegpoints)), key_target_tensor: (2*nKeypoints, sum_batch(nSegpoints))

    # change last dim to 2 for piecewise distance
    xy_pred = torch.stack((key_tensor[:nKeypoints].view(nKeypoints * roi_ind.size(0)),
                           key_tensor[nKeypoints:2*nKeypoints].view(nKeypoints * roi_ind.size(0))), dim=1)
    xy_gt = torch.stack((xy_gt[:nKeypoints].view(nKeypoints * roi_ind.size(0)),
                         xy_gt[nKeypoints:2*nKeypoints].view(nKeypoints * roi_ind.size(0))), dim=1)
    conf_pred = key_tensor[2*nKeypoints:].view(nKeypoints * roi_ind.size(0))

    L1loss = nn.L1Loss()
    pos_loss = L1loss(xy_pred, xy_gt) #/ nWidth

    pdist = nn.PairwiseDistance(p=2)
    conf_loss = L1loss(conf_pred, torch.exp(-tau * pdist(xy_pred, xy_gt)).detach())

    return norm_factor * (beta * pos_loss + gamma * conf_loss)


if __name__ == "__main__":
    data_root = args.data_root

    CUDA = 1 # args.gpu is not None
    GPU_ID = [0, 1]

    if args.edgemap:
        edgemap = True
    else:
        edgemap = False

    train_dataset = LFDataset(root_path=data_root, edgemap=edgemap)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=6)

    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS, keypoints=NUM_KEYPOINTS).cuda()
        if args.partial_preload:
            model.load_segonly_state_dict(torch.load(args.partial_preload))
        model = torch.nn.DataParallel(model, GPU_ID).cuda()
        class_weights = 1.0/train_dataset.get_class_probability().cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

        # class_weights = 1.0/train_dataset.get_class_probability()
        criterion = torch.nn.CrossEntropyLoss()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)

    train()

    print('train success')
