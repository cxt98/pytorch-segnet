"""
Infer segmentation results from a trained SegNet model


Usage:
python inference.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                    --val_path ImageSets/Segmentation/val.txt \
                    --img_dir JPEGImages \
                    --mask_dir SegmentationClass \
                    --model_path /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                    --output_dir /home/SharedData/intern_sayan/PascalVOC2012/predictions \
                    --gpu 1

new lf usage 
python ./src/inference.py --data_root /home/cxt/Documents/research/lf_dope/lf_trans_dataset/synthetic/ --model_path /home/cxt/Documents/research/lf_dope/pytorch-segnet/checkpoint/ --output_dir /home/cxt/Documents/research/lf_dope/pytorch-segnet/output/

"""


from __future__ import print_function
import argparse
from dataset import LFDataset, NUM_CLASSES
import matplotlib.pyplot as plt
from model import SegNet
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

plt.switch_backend('agg')
plt.axis('off')


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES

BATCH_SIZE = 1


# Arguments
parser = argparse.ArgumentParser(description='Validate a SegNet model')

parser.add_argument('--data_root', required=True)
# parser.add_argument('--val_path', required=True)
# parser.add_argument('--img_dir', required=True)
# parser.add_argument('--mask_dir', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--output_dir', required=True)
# parser.add_argument('--gpu', type=int)

args = parser.parse_args()



def validate():
    model.eval()

    for batch_idx, batch in enumerate(val_dataloader):
        input_tensor = torch.autograd.Variable(batch['image'])
        target_tensor = torch.autograd.Variable(batch['mask'])

        # if CUDA:
        #     input_tensor = input_tensor.cuda()
        #     target_tensor = target_tensor.cuda()

        predicted_tensor, softmaxed_tensor = model(input_tensor)
        # loss = criterion(predicted_tensor, target_tensor)
        
        for idx, predicted_mask in enumerate(softmaxed_tensor):
            target_mask = target_tensor[idx]
            input_image = input_tensor[idx]

            fig = plt.figure()

            a = fig.add_subplot(1,3,1)
            plt.imshow(input_image.transpose(0, 2).transpose(0, 1))
            a.set_title('Input Image')

            a = fig.add_subplot(1,3,2)
            predicted_mx = predicted_mask.data.cpu().numpy()
            predicted_mx = predicted_mx.argmax(axis=0)
            plt.imshow(predicted_mx)
            a.set_title('Predicted Mask')

            a = fig.add_subplot(1,3,3)
            target_mx = target_mask.data.cpu().numpy() * 255
            Image.fromarray(target_mx.astype(np.uint8)).save(str(idx) + '.png')
            plt.imshow(target_mx)
            a.set_title('Ground Truth')

            fig.savefig(os.path.join(OUTPUT_DIR, "prediction_{}_{}.png".format(batch_idx, idx)))

            plt.close(fig)


if __name__ == "__main__":
    data_root = args.data_root
    SAVED_MODEL_PATH = args.model_path + 'model_best.pth'
    OUTPUT_DIR = args.output_dir

    CUDA = 1
    GPU_ID = [0]

    val_dataset = LFDataset(root_path=data_root)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=4)


    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda()
        model = torch.nn.DataParallel(model, GPU_ID).cuda() 
        # class_weights = 1.0/val_dataset.get_class_probability().cuda(GPU_ID)
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

        # class_weights = 1.0/val_dataset.get_class_probability()
        criterion = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    
    validate()