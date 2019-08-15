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

LF_CLASSES = ('background',  # always index 0
              'glass')

NUM_CLASSES = len(LF_CLASSES)


class LFDataset(Dataset):
    def __init__(self, root_path, transform=None, validation=False):
        self.images = []
        self.masks = []
        self.transform = transform
        self.validation = validation
        self.angular_size = 5

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
            data = {
                'image': torch.FloatTensor(image),
                'mask': torch.LongTensor(gt_mask)
            }
        else:
            data = {
                'image': torch.FloatTensor(image)
            }

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for mask_path in self.masks:

            raw_image = Image.open(mask_path)
            imx_t = np.array(raw_image)
            imx_t[imx_t < 255] = 0
            imx_t[imx_t == 255] = 1
            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

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
                    if exists(maskpath):
                        self.masks.append(maskpath)
                        self.images.append(lfpath)

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
        imx_t[imx_t < 255] = 0
        imx_t[imx_t == 255] = 1
        return imx_t


class PascalLFDataset(Dataset):
    """Pascal LF 2007 Dataset"""

    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {
            'image': torch.FloatTensor(image),
            'mask': torch.LongTensor(gt_mask)
        }

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

            raw_image = Image.open(mask_path).resize((224, 224))
            imx_t = np.array(raw_image).reshape(224 * 224)
            imx_t[imx_t == 255] = NUM_CLASSES

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.transpose(raw_image.resize((224, 224)), (2, 1, 0))
        imx_t = np.array(raw_image, dtype=np.float32) / 255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        imx_t = np.array(raw_image)
        # border
        imx_t[imx_t == 255] = NUM_CLASSES

        return imx_t


if __name__ == "__main__":
    data_root = os.path.join("data", "LFdevkit", "LF2007")
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    img_dir = os.path.join(data_root, "JPEGImages")
    mask_dir = os.path.join(data_root, "SegmentationObject")

    objects_dataset = PascalLFDataset(list_file=list_file_path,
                                      img_dir=img_dir,
                                      mask_dir=mask_dir)

    print(objects_dataset.get_class_probability())

    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']

    image.transpose_(0, 2)

    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow(image)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)

    plt.show()
