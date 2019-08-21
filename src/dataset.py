"""LF Dataset Segmentation Dataloader"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import glob
from os.path import exists
import json

LF_CLASSES = {'background':0,  # always index 0
              'wine_cup':1}

NUM_CLASSES = len(LF_CLASSES)


class LFDataset(Dataset):
    def __init__(self, root_path, transform=None, validation=False, edgemap=False):
        self.images = []
        self.masks = []
        self.jsonfile = []
        self.transform = transform
        self.validation = validation
        self.angular_size = 5
        self.img_size = 224
        self.edgemap = edgemap
        self.root_path = root_path

        if self.validation:
            self.findallimg_validate(self.root_path)
        else:
            self.findallimg_train(self.root_path)  # save paths of all images, masks to self.images, self.masks
            self.counts = self.__compute_class_probability()
        print("Load data complete, image number: {}" .format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.load_image(path=self.images[index])
        if not self.validation:
            gt_mask = self.load_mask(path=self.masks[index])
            json_info = self.loadjson(path=self.jsonfile[index])
            gt_offset = self.generate_offset_map(jsoninfo=json_info, mask=gt_mask)
            data = {
                'image': torch.FloatTensor(image),
                'mask': torch.LongTensor(gt_mask),
                'gt_offset': torch.FloatTensor(gt_offset)
            }
        else:
            data = {
                'image': torch.FloatTensor(image)
            }

        return data

    def loadjson(self,path):
        """
        Loads the data from a json file.
        If there are no objects of interest, then load all the objects.
        """
        with open(path) as data_file:
            data = json.load(data_file)
        # print (path)

        points_keypoints_2d = []
        pointsBoxes = []
        boxes = []
        class_name = []
        # points_keypoints_3d = []
        # pointsBelief = []
        # poses = []
        # centroids = []
        #
        # translations = []
        # rotations = []
        # points = []

        for i_line in range(len(data['objects'])):
            info = data['objects'][i_line]
            class_name.append(info['class'])
            box = info['bounding_box']
            boxToAdd = []

            boxToAdd.append(float(box['top_left'][0]))
            boxToAdd.append(float(box['top_left'][1]))
            boxToAdd.append(float(box["bottom_right"][0]))
            boxToAdd.append(float(box['bottom_right'][1]))
            boxes.append(boxToAdd)

            boxpoint = [(boxToAdd[1], boxToAdd[0]), (boxToAdd[3], boxToAdd[0]),
                        (boxToAdd[1], boxToAdd[2]), (boxToAdd[3], boxToAdd[2])]   # use x,y to index the bbox

            pointsBoxes.append(boxpoint)

            # 2d projected key points
            point2dToAdd = []
            pointdata = info['projected_cuboid']
            for p in pointdata:
                point2dToAdd.append((p[0], p[1]))   # change x,y index to row,col index

            # Get the centroids
            pcenter = info['projected_cuboid_centroid']

            point2dToAdd.append((pcenter[0], pcenter[1]))
            points_keypoints_2d.append(point2dToAdd)

        return {
            "class": class_name,
            "bbox": pointsBoxes,
            "keypoints_2d": points_keypoints_2d # 8 keypoints + center
        }
    def generate_offset_map(self,jsoninfo,mask):
        temp_mask = np.copy(mask)
        img_size = mask.shape
        temp_mask[temp_mask == NUM_CLASSES] = 0  # edge to background
        # temp_mask[temp_mask > 0] = 1 # create a mask to show where objects exist
        x_offset = np.tile(np.arange(img_size[1]),(img_size[0],1))
        y_offset = np.transpose(np.copy(x_offset))
        x_offset[temp_mask==0] = 0
        y_offset[temp_mask==0] = 0
        x_offset = np.repeat(x_offset[:, :, np.newaxis], 9, axis=2)
        y_offset = np.repeat(y_offset[:, :, np.newaxis], 9, axis=2)
        x_mask = np.zeros([img_size[0], img_size[1], 9])
        y_mask = np.zeros([img_size[0], img_size[1], 9])
        for idx in range(len(jsoninfo['class'])):
            class_name = jsoninfo['class'][idx]
            bbox = jsoninfo['bbox'][idx]
            keypoint2d = jsoninfo['keypoints_2d'][idx]
            target_label = LF_CLASSES[class_name]
            current_instance_mask = np.full(img_size,False)
            bbox_rowindex = (self.boundray_check(bbox[0][1], 0, img_size[0]),
                             self.boundray_check(bbox[2][1], 0, img_size[0]))
            bbox_colindex = (self.boundray_check(bbox[0][0], 0, img_size[1]),
                             self.boundray_check(bbox[1][0], 0, img_size[1]))
            if bbox_rowindex[0] == bbox_rowindex[1] or bbox_colindex[0] == bbox_colindex[1]:
                continue
            else:
                for kp in range(len(keypoint2d)):
                    current_instance_mask[bbox_rowindex[0]:bbox_rowindex[1],bbox_colindex[0]:bbox_colindex[1]] = \
                        (temp_mask[bbox_rowindex[0]:bbox_rowindex[1], bbox_colindex[0]:bbox_colindex[1]]==target_label)

                    x_mask[:, :, kp] = x_mask[:, :, kp] + np.multiply(keypoint2d[kp][0] * current_instance_mask.astype(int),
                                                                      (x_mask[:, :, kp] == 0).astype(int))

                    y_mask[:, :, kp] = y_mask[:, :, kp] + np.multiply(keypoint2d[kp][0] * current_instance_mask.astype(int),
                                                                        (y_mask[:, :, kp] == 0).astype(int))

        x_offset = (x_offset - x_mask) / img_size[1]
        y_offset = (y_offset - y_mask) / img_size[0]
        self.save_offset_mask_to_img("/media/alienicp/5f3d1485-53ed-4161-af42-63cef2fc27a1/home/logan/LFdata/wc/segmentation/test",x_offset,"x")
        self.save_offset_mask_to_img(
            "/media/alienicp/5f3d1485-53ed-4161-af42-63cef2fc27a1/home/logan/LFdata/wc/segmentation/test", y_offset,
            "y")
        return np.concatenate((x_offset,y_offset),axis=2)


    def save_offset_mask_to_img(self,path, mask,save_prefix):
        for i in range(mask.shape[2]):
            mask[:,:,i] = (mask[:,:,i] - mask.min()) / (mask.max() - mask.min())
            im = Image.fromarray(np.uint8(mask[:,:,i]*255))
            im.save(os.path.join(path,"{}_offsetmap_{}.png".format(save_prefix,i)))

    def boundray_check(self,value,min,max):

        value = int(round(min)) if value < int(round(min)) else int(round(value))
        value = int(round(max)) if value > int(round(max)) else int(round(value))

        return value


    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES + 1))

        for mask_path in self.masks:

            raw_image = Image.open(mask_path)
            imx_t = np.array(raw_image)
            # imx_t[imx_t < 255] = 0
            # imx_t[imx_t == 255] = 1
            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)
            counts[NUM_CLASSES] = counts[0]/50  # weight of boundary around objects, set 50x of background

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / float(np.sum(values))

        return torch.Tensor(p_values)

    def findallimg_validate(self, path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path)
                   if os.path.isdir(os.path.join(path, o))]
        if len(folders) > 0:
            for path_entry in folders:
                self.findallimg_validate(path_entry)
        else:
            for imgpath in sorted(glob.glob(path + '/*LF.jpg')):
                self.images.append(imgpath)

    def findallimg_train(self, path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path)
                   if os.path.isdir(os.path.join(path, o))]
        if len(folders) > 0:
            for path_entry in folders:
                self.findallimg_train(path_entry)
        else:
            if len(glob.glob(path + "/*.Sub3_3.png")) > 0:
                # FIRST TIME: exists subaperture images: combine to light field images then remove subapertures
                for subpath in sorted(glob.glob(path + "/*.Sub3_3.png")):
                    if self.edgemap:
                        maskpath = subpath.replace('.png', '.cs.edge.png')
                    else:
                        maskpath = subpath.replace('.png', '.cs.png')
                    if exists(maskpath):
                        self.masks.append(maskpath)
                        lfpath = subpath.replace('.png', '.lf.png')
                        if not exists(lfpath):  # combine all subapertures to get light field image
                            if self.combine_img(subpath) == 0:
                                self.images.append(lfpath)
                            else:
                                self.masks.remove(maskpath)
                        else:
                            self.images.append(lfpath)
                        for i in range(self.angular_size):  # remove original subaperture images, only run once
                            for j in range(self.angular_size):
                                os.remove(subpath.replace('3_3', str(i + 1) + '_' + str(j + 1)))
            else:
                # NEXT TIMES: directly get lf imgs and masks
                for lfpath in sorted(glob.glob(path + "/*.Sub3_3.lf.png")):
                    maskpath = lfpath.replace('lf', 'cs')
                    jsonpath = lfpath.replace("lf.png","json")
                    if exists(maskpath):
                        self.masks.append(maskpath)
                        self.images.append(lfpath)
                        self.jsonfile.append(jsonpath)


    def combine_img(self, path):
        # first check all subaperture imgs exist, then combine them and save as one lf image
        img = Image.open(path)
        lf_img = np.zeros(shape=(img.size[1] * self.angular_size, img.size[0] * self.angular_size, 3), dtype=np.uint8)
        imgs = []
        for i in range(self.angular_size):
            for j in range(self.angular_size):
                imgpath = path.replace('3_3', str(i + 1) + '_' + str(j + 1))
                if not exists(imgpath):
                    print('error: not exist ' + imgpath + ' as subaperture image')
                    return 1
                else:
                    imgs.append(np.asarray(Image.open(imgpath))[:, :, :3])
        for i in range(self.angular_size):
            for j in range(self.angular_size):
                lf_img[i::5, j::5, :] = imgs[i * 5 + j]
        Image.fromarray(lf_img).save(path.replace('.png', '.lf.png'))
        return 0

    def load_image(self, path=None):
        raw_image = Image.open(path)
        try:
            raw_image = np.transpose(raw_image, (2, 0, 1))
            raw_image_3d = np.zeros((raw_image.shape[0], self.angular_size ** 2,
                                     raw_image.shape[1] / self.angular_size,
                                     raw_image.shape[2] / self.angular_size))
            for k in range(3):  # divide r, g, b channels
                for i in range(self.angular_size):
                    for j in range(self.angular_size):
                        raw_image_3d[k, i * self.angular_size + j] = raw_image[k, i::5, j::5]

        except:
            os.remove(path)
            self.combine_img(path)
            print(path)
        imx_t = np.array(raw_image_3d, dtype=np.float32) / 255.0
        # imx_t = np.array(raw_image, dtype=np.float32) / 255.0
        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)  # .convert('RGB')
        # raw_image = np.transpose(raw_image, (2, 1, 0))
        # raw_image = raw_image.resize((224, 224))
        imx_t = np.array(raw_image)
        # border
        imx_t[imx_t == 255] = NUM_CLASSES

        return imx_t



