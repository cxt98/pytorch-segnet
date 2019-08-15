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

# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES

NUM_EPOCHS = 100

LEARNING_RATE = 1e-5
BATCH_SIZE = 16


# Arguments
parser = argparse.ArgumentParser(description='Train a SegNet model')

parser.add_argument('--data_root', required=True)
# parser.add_argument('--train_path', required=True)
# parser.add_argument('--img_dir', required=True)
# parser.add_argument('--mask_dir', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--checkpoint')
# parser.add_argument('--gpu', type=int)

args = parser.parse_args()



def train():
    is_better = True
    prev_loss = float('inf')

    model.train()
    print ("Batch Number:" + str(len(train_dataloader)))
    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()
        batch_id = 0
        for batch in train_dataloader:
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])

            if CUDA:
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()

            predicted_tensor, softmaxed_tensor = model(input_tensor)


            optimizer.zero_grad()
            loss = criterion(softmaxed_tensor, target_tensor)
            loss.backward()
            optimizer.step()

            if batch_id % 20 == 0:
                print("Epoch #{}\tBatch #{}\tLoss: {:.8f}".format(epoch + 1, batch_id, loss))
            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()
            batch_id = batch_id + 1

        loss_f = loss_f / len(train_dataloader)
        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pth"))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_" + str(epoch) + ".pth"))
        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))


if __name__ == "__main__":
    data_root = args.data_root

    CUDA = 1 # args.gpu is not None
    GPU_ID = [0,1]


    train_dataset = LFDataset(root_path=data_root)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=6)



    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda()
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
        


    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)


    train()


    print('train over')
