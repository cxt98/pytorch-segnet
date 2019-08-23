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
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

plt.switch_backend('agg')
plt.axis('off')


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES + 1

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

g_img = None
g_draw = None

Debug = False
targetLabel = 1 # 1 for glass

colorcodes = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b']

def validate():
    model.eval()

    for batch_idx, batch in enumerate(val_dataloader):
        input_tensor = torch.autograd.Variable(batch['image'])

        # gt_kp = batch['keypoints_2d'].data.cpu().numpy() ########### Only for debug use, plot the gt against estimation


        xseg_output, xkey_output = model(input_tensor)
        # loss = criterion(predicted_tensor, target_tensor)
        img_size = (xkey_output.shape[2],xkey_output.shape[3])
        x_offsetmask = np.tile(np.arange(img_size[1]),(img_size[0],1))
        y_offsetmask = np.transpose(np.tile(np.arange(img_size[0]),(img_size[1],1)))
        x_offsetmask = np.repeat(x_offsetmask[:, :, np.newaxis], 9, axis=2)
        y_offsetmask = np.repeat(y_offsetmask[:, :, np.newaxis], 9, axis=2)

        for idx, predicted_keypoints in enumerate(xkey_output):
            save_to_npy = np.zeros([img_size[0],img_size[1],3,9])
            input_image = input_tensor[idx]
            seg_mask = xseg_output[idx].data.cpu().numpy()
            seg_mask = seg_mask.argmax(axis=0)
            predicted_keypoints = torch.squeeze(predicted_keypoints)
            predicted_kps = predicted_keypoints.data.cpu().numpy()
            predicted_kpX = x_offsetmask.transpose(2,0,1) - predicted_kps[0:9, :, :] * img_size[1]
            predicted_kpY = y_offsetmask.transpose(2,0,1) - predicted_kps[9:18, :, :] * img_size[0]
            predicted_conf = predicted_kps[18::, :, :]

            for corner_idx in range(predicted_kpX.shape[0]):
                if Debug:
                    kpX = predicted_kpX[corner_idx, :, :]
                    kpY = predicted_kpY[corner_idx, :, :]
                    kpC = predicted_conf[corner_idx, :, :]
                    kpX[seg_mask != targetLabel] = 0
                    kpY[seg_mask != targetLabel] = 0
                    kpC[seg_mask != targetLabel] = 0
                    save_to_npy[:, :, :, corner_idx] = np.dstack((kpX,kpY,kpC))

                else:
                    fig = plt.figure()
                    plt.imshow(input_image[:, 13].transpose(0, 2).transpose(0, 1))
                    kpX = predicted_kpX[corner_idx, seg_mask == targetLabel]
                    kpY = predicted_kpY[corner_idx, seg_mask == targetLabel]
                    kpC = predicted_conf[corner_idx, seg_mask == targetLabel]
                    kpX = np.reshape(kpX, (1, -1))
                    kpY = np.reshape(kpY, (1, -1))
                    kpC = np.reshape(kpC, (1, -1))
                    invalid_mask = np.logical_and(np.logical_and(kpX >= 0, kpX <= img_size[1]),
                                                  np.logical_and(kpY >= 0, kpY <= img_size[0]))
                    kpX = np.ma.MaskedArray(kpX, mask=~invalid_mask)
                    kpY = np.ma.MaskedArray(kpY, mask=~invalid_mask)
                    kpC = np.ma.MaskedArray(kpC, mask=~invalid_mask)
                    kpX = np.ma.compress_cols(kpX)
                    kpY = np.ma.compress_cols(kpY)
                    kpC = np.ma.compress_cols(kpC)
                    topN = 100
                    kp = np.hstack((np.transpose(kpX), np.transpose(kpY), np.transpose(kpC)))
                    kp_topN = sorted(kp, key=lambda entry: entry[-1], reverse=True)[:topN]
                    kpX_topN = [a[0] for a in kp_topN]
                    kpY_topN = [a[1] for a in kp_topN]

                    plt.plot(kpX_topN, kpY_topN, colorcodes[corner_idx] + 'o')
                    plt.show()
                    ########### Only for debug use, plot the gt against estimation
                    # for cup_number in range(gt_kp.shape[1]):
                    #     plt.plot(gt_kp[0,cup_number,corner_idx,0], gt_kp[0,cup_number,corner_idx,1],
                    #              'rx', linewidth = 4,markersize = 10)
                    #     plt.show()
                    #
                    # fig.savefig(os.path.join(OUTPUT_DIR, "prediction_kp_{}_{}_all.png".format(batch_idx,corner_idx)))
                    # plt.close(fig)
            # if Debug:
            #     np.save(os.path.join(OUTPUT_DIR, "prediction_{}_segout".format(batch_idx)), save_to_npy)
            # vertices_xy = get_bbox_vertices(save_to_npy)
            # img = draw_bbox(vertices_xy, input_image[:, 13].transpose(0, 2).transpose(0, 1))
            # img.save(os.path.join(OUTPUT_DIR, "prediction_bbox_{}.png".format(batch_idx)))

        if not Debug:
            for idx, predicted_mask in enumerate(xseg_output):
                input_image = input_tensor[idx]

                fig = plt.figure()

                a = fig.add_subplot(1,2,1)
                plt.imshow(input_image[:,13].transpose(0, 2).transpose(0, 1)) # extract CenterView from 3D LF input
                a.set_title('Input Image')

                a = fig.add_subplot(1,2,2)
                predicted_mx = predicted_mask.data.cpu().numpy()
                predicted_mx = predicted_mx.argmax(axis=0)
                # for display
                predicted_mx[predicted_mx == 1] = 128
                predicted_mx[predicted_mx == 2] = 255
                plt.imshow(predicted_mx)
                a.set_title('Predicted Mask')

                # a = fig.add_subplot(1,3,3)
                # target_mx = target_mask.data.cpu().numpy() * 255
                # Image.fromarray(target_mx.astype(np.uint8)).save(str(idx) + '.png')
                # plt.imshow(target_mx)
                # a.set_title('Ground Truth')

                fig.savefig(os.path.join(OUTPUT_DIR, "prediction_{}_segout.png".format(batch_idx)))
                print("Predicted {}th frame".format(batch_idx))
                plt.close(fig)




            # processes to save 4d output "combined_map", "img" should be original image

            # vertices_xy = get_bbox_vertices(combined_map)
            # img = draw_bbox(vertices_xy, input_image[:,13])
            # img.save("bbox_{}.png".format(batch_idx))


def get_bbox_vertices(combined_map):
    # combined map format: width(224) * height(224) * 3(offset_x, offset_y, conf) * nKeypoints(9)
    # use weighted mean of top N estimates with highest confidence
    topN = 100
    row, col = np.nonzero(combined_map[:, :, 0, 0])
    vertices_xy = []
    for i in range(combined_map.shape[-1]):
        vertices_candidate = np.transpose(np.vstack((combined_map[row, col, 0, i] + row, combined_map[row, col, 1, i] + col, combined_map[row, col, 2, i])))
        sorted_candidate = sorted(vertices_candidate, key=lambda entry: entry[-1], reverse=True)
        vertices_xy.append((np.average(sorted_candidate[:topN][0], weights=sorted_candidate[:topN][-1]),
                           np.average(sorted_candidate[:topN][1], weights=sorted_candidate[:topN][-1])))
    return vertices_xy


def draw_bbox(vertices_xy, img_np, color=(255, 0, 0)):

    def DrawLine(point1, point2, lineColor, lineWidth):
        '''Draws line on image'''
        global g_draw
        if not point1 is None and point2 is not None:
            g_draw.line([point1, point2], fill=lineColor, width=lineWidth)

    def DrawDot(point, pointColor, pointRadius):
        '''Draws dot (filled circle) on image'''
        global g_draw
        if point is not None:
            xy = [
                point[0] - pointRadius,
                point[1] - pointRadius,
                point[0] + pointRadius,
                point[1] + pointRadius
            ]
            g_draw.ellipse(xy,
                           fill=pointColor,
                           outline=pointColor
                           )

    def DrawCube(points, color=(255, 0, 0)):
        '''
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        '''

        lineWidthForDrawing = 2

        # draw front
        DrawLine(points[0], points[1], color, lineWidthForDrawing)
        DrawLine(points[1], points[2], color, lineWidthForDrawing)
        DrawLine(points[3], points[2], color, lineWidthForDrawing)
        DrawLine(points[3], points[0], color, lineWidthForDrawing)

        # draw back
        DrawLine(points[4], points[5], color, lineWidthForDrawing)
        DrawLine(points[6], points[5], color, lineWidthForDrawing)
        DrawLine(points[6], points[7], color, lineWidthForDrawing)
        DrawLine(points[4], points[7], color, lineWidthForDrawing)

        # draw sides
        DrawLine(points[0], points[4], color, lineWidthForDrawing)
        DrawLine(points[7], points[3], color, lineWidthForDrawing)
        DrawLine(points[5], points[1], color, lineWidthForDrawing)
        DrawLine(points[2], points[6], color, lineWidthForDrawing)

        # draw dots
        DrawDot(points[0], pointColor=color, pointRadius=4)
        DrawDot(points[1], pointColor=color, pointRadius=4)

        # draw x on the top
        DrawLine(points[0], points[5], color, lineWidthForDrawing)
        DrawLine(points[1], points[4], color, lineWidthForDrawing)

    global g_img
    global g_draw
    img = Image.fromarray((img_np.numpy() * 255).astype('uint8'))
    g_draw = ImageDraw.Draw(img)
    DrawCube(vertices_xy)
    return img


if __name__ == "__main__":
    # test code for selecting keypoints and visualization
    # img = Image.open("/home/cxt/Documents/research/lf_dope/pytorch-segnet/test_for_training/000000.Sub3_3.lf.png")
    # combined_map = np.load("/home/cxt/Documents/research/lf_dope/pytorch-segnet/test_for_training/prediction_0_segout.png.npy")
    # vertices_xy = get_bbox_vertices(combined_map)
    # img = draw_bbox(vertices_xy, img)
    # img.save("bbox_test.png")
    ###############

    data_root = args.data_root
    SAVED_MODEL_PATH = args.model_path
    OUTPUT_DIR = args.output_dir

    CUDA = 1
    GPU_ID = [0,1]

    val_dataset = LFDataset(root_path=data_root, validation=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=4)


    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS, keypoints=9).cuda()
        model = torch.nn.DataParallel(model, GPU_ID).cuda()
        # class_weights = 1.0 / val_dataset.get_class_probability().cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

        # class_weights = 1.0/val_dataset.get_class_probability()
        criterion = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(SAVED_MODEL_PATH))

    validate()