"""
Pytorch implementation of SegNet (https://arxiv.org/pdf/1511.00561.pdf)
"""

from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import pprint

F = nn.functional
DEBUG = False

vgg16_dims = [
    (64, 64, 'M'),  # Stage - 1
    (128, 128, 'M'),  # Stage - 2
    (256, 256, 256, 'M'),  # Stage - 3
    (512, 512, 512, 'M'),  # Stage - 4
    (512, 512, 512, 'M')  # Stage - 5
]

decoder_dims = [
    ('U', 512, 512, 512),  # Stage - 5
    ('U', 512, 512, 512),  # Stage - 4
    ('U', 256, 256, 256),  # Stage - 3
    ('U', 128, 128),  # Stage - 2
    ('U', 64, 64)  # Stage - 1
]


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels, keypoints):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_channels = input_channels

        self.angular_size = 5
        self.keypoints = keypoints

        self.createFeatureLayers()
        self.createSegLayers()
        self.createRegLayers()
        # self.freeze_seg()
        # self.freeze_encoder()

    def createFeatureLayers(self):
        self.vgg16 = models.vgg16(pretrained=True)

        # add angular filter layers specific for 3D light field image inputs before feature extraction

        self.angular_conv = nn.Sequential(*[
            nn.Conv3d(in_channels=self.input_channels,
                      out_channels=64,
                      kernel_size=(self.angular_size ** 2, 1, 1),
                      stride=(self.angular_size ** 2, 1, 1),
                      padding=(0, 0, 0)),
            nn.BatchNorm3d(64)
        ])

        self.EPI_row = nn.Sequential(*[
            nn.Conv3d(in_channels=self.input_channels,
                      out_channels=64,
                      kernel_size=(self.angular_size, 3, 3),
                      stride=(self.angular_size, 1, 1),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(64)
        ])


        self.EPI_col = nn.Sequential(*[
            nn.Conv3d(in_channels=self.input_channels,
                      out_channels=64,
                      kernel_size=(self.angular_size, 3, 3),
                      stride=(1, 1, 1),
                      dilation=(self.angular_size, 1, 1),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(64)
        ])


        self.encoder_conv_00 = nn.Sequential(*[
            nn.Conv2d(in_channels=64 * (2 * self.angular_size + 1),
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])


        self.encoder_conv_01 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])



        self.encoder_conv_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])



        self.encoder_conv_11 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])



        self.encoder_conv_20 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])



        self.encoder_conv_21 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])



        self.encoder_conv_22 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])



        self.encoder_conv_30 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])



        self.encoder_conv_31 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])



        self.encoder_conv_32 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])



        self.encoder_conv_40 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])



        self.encoder_conv_41 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])



        self.encoder_conv_42 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])



        self.init_vgg_weights()

    def createSegLayers(self):
        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])



        self.decoder_convtr_41 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])



        self.decoder_convtr_40 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])



        self.decoder_convtr_32 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])



        self.decoder_convtr_31 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])



        self.decoder_convtr_30 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])



        self.decoder_convtr_22 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])



        self.decoder_convtr_21 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])



        self.decoder_convtr_20 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128)
        ])



        self.decoder_convtr_11 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128)
        ])



        self.decoder_convtr_10 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64)
        ])



        self.decoder_convtr_01 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64)
        ])



        self.decoder_convtr_00 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=self.output_channels,
                               kernel_size=3,
                               padding=1)
        ])


    def createRegLayers(self):
        # Decoder layers

        self.decoder_convtr_42_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_41_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_40_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_32_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_31_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_30_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_22_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_21_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_20_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128)
        ])
        self.decoder_convtr_11_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128)
        ])
        self.decoder_convtr_10_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64)
        ])
        self.decoder_convtr_01_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64)
        ])
        self.decoder_convtr_00_k = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=3*self.keypoints,
                               kernel_size=3,
                               padding=1)
        ])

    def forward(self, input_img):
        """
        Forward pass `input_img` through the network
        """

        # Encoder - angular filter layer
        dim_ang = input_img.size()
        x_ang0 = self.angular_conv(input_img)
        x_EPI_row = self.EPI_row(input_img)
        x_EPI_col = self.EPI_col(input_img)

        concatenate_feature = torch.cat((x_ang0, x_EPI_row, x_EPI_col), dim=2)
        x_concat = F.relu(concatenate_feature)

        x_concat = x_concat.view(x_concat.shape[0], x_concat.shape[1] * x_concat.shape[2], x_concat.shape[3],
                                 x_concat.shape[4])
        # Encoder Stage - 1
        # dim_0 = [input_img.size()[0]/5, input_img.size()[1]/5]
        dim_0 = input_img.size()
        dim_0 = torch.Size([dim_0[0], dim_0[1], dim_0[3], dim_0[4]])

        x_00 = F.relu(self.encoder_conv_00(x_concat))
        x_01 = F.relu(self.encoder_conv_01(x_00))
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        x_42 = F.relu(self.encoder_conv_42(x_41))
        x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)

        # Decoder

        dim_d = x_4.size()

        # Decoder Stage - 5
        x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d = F.relu(self.decoder_convtr_42(x_4d))
        x_41d = F.relu(self.decoder_convtr_41(x_42d))
        x_40d = F.relu(self.decoder_convtr_40(x_41d))
        dim_4d = x_40d.size()

        # Decoder Stage - 4
        x_3d = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = F.relu(self.decoder_convtr_30(x_31d))
        dim_3d = x_30d.size()

        # Decoder Stage - 3
        x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))
        dim_2d = x_20d.size()

        # Decoder Stage - 2
        x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = F.relu(self.decoder_convtr_10(x_11d))
        dim_1d = x_10d.size()

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)
        dim_0d = x_00d.size()

        xseg_output = F.softmax(x_00d, dim=1)

        # Another decoder branch for regressing keypoints

        # Decoder Stage - 5
        x_4d_k = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d_k = F.relu(self.decoder_convtr_42_k(x_4d_k))
        x_41d_k = F.relu(self.decoder_convtr_41_k(x_42d_k))
        x_40d_k = F.relu(self.decoder_convtr_40_k(x_41d_k))
        dim_4d_k = x_40d_k.size()

        # Decoder Stage - 4
        x_3d_k = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d_k = F.relu(self.decoder_convtr_32_k(x_3d_k))
        x_31d_k = F.relu(self.decoder_convtr_31_k(x_32d_k))
        x_30d_k = F.relu(self.decoder_convtr_30_k(x_31d_k))
        dim_3d_k = x_30d_k.size()

        # Decoder Stage - 3
        x_2d_k = F.max_unpool2d(x_30d_k, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d_k = F.relu(self.decoder_convtr_22_k(x_2d_k))
        x_21d_k = F.relu(self.decoder_convtr_21_k(x_22d_k))
        x_20d_k = F.relu(self.decoder_convtr_20_k(x_21d_k))
        dim_2d_k = x_20d_k.size()

        # Decoder Stage - 2
        x_1d_k = F.max_unpool2d(x_20d_k, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d_k = F.relu(self.decoder_convtr_11_k(x_1d_k))
        x_10d_k = F.relu(self.decoder_convtr_10_k(x_11d_k))
        dim_1d_k = x_10d_k.size()

        # Decoder Stage - 1
        x_0d_k = F.max_unpool2d(x_10d_k, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d_k = F.relu(self.decoder_convtr_01_k(x_0d_k))
        x_00d_k = self.decoder_convtr_00_k(x_01d_k)
        dim_0d_k = x_00d_k.size()

        # xkey_softmax = F.softmax(x_00d_k, dim=self.keypoints * 3)
        xkey_output = x_00d_k
        xkey_output[:, 2*self.keypoints:] = F.sigmoid(xkey_output[:, 2*self.keypoints:])

        if DEBUG:
            print("dim_ang: {}".format(dim_ang))
            print("dim_0: {}".format(dim_0))
            print("dim_1: {}".format(dim_1))
            print("dim_2: {}".format(dim_2))
            print("dim_3: {}".format(dim_3))
            print("dim_4: {}".format(dim_4))

            print("dim_d: {}".format(dim_d))
            print("dim_4d: {}".format(dim_4d))
            print("dim_3d: {}".format(dim_3d))
            print("dim_2d: {}".format(dim_2d))
            print("dim_1d: {}".format(dim_1d))
            print("dim_0d: {}".format(dim_0d))
            print("dim_4d_k: {}".format(dim_4d_k))
            print("dim_3d_k: {}".format(dim_3d_k))
            print("dim_2d_k: {}".format(dim_2d_k))
            print("dim_1d_k: {}".format(dim_1d_k))
            print("dim_0d_k: {}".format(dim_0d_k))

        return xseg_output, xkey_output

    def freeze_encoder(self):
        self.angular_conv[0].weight.requires_grad = False
        self.angular_conv[0].bias.requires_grad = False
        self.EPI_row[0].weight.requires_grad = False
        self.EPI_row[0].bias.requires_grad = False
        self.EPI_col[0].weight.requires_grad = False
        self.EPI_col[0].bias.requires_grad = False
        self.encoder_conv_00[0].weight.requires_grad = False
        self.encoder_conv_00[0].bias.requires_grad = False
        self.encoder_conv_01[0].weight.requires_grad = False
        self.encoder_conv_01[0].bias.requires_grad = False
        self.encoder_conv_10[0].weight.requires_grad = False
        self.encoder_conv_10[0].bias.requires_grad = False
        self.encoder_conv_11[0].weight.requires_grad = False
        self.encoder_conv_11[0].bias.requires_grad = False
        self.encoder_conv_20[0].weight.requires_grad = False
        self.encoder_conv_20[0].bias.requires_grad = False
        self.encoder_conv_21[0].weight.requires_grad = False
        self.encoder_conv_21[0].bias.requires_grad = False
        self.encoder_conv_22[0].weight.requires_grad = False
        self.encoder_conv_22[0].bias.requires_grad = False
        self.encoder_conv_30[0].weight.requires_grad = False
        self.encoder_conv_30[0].bias.requires_grad = False
        self.encoder_conv_31[0].weight.requires_grad = False
        self.encoder_conv_31[0].bias.requires_grad = False
        self.encoder_conv_32[0].weight.requires_grad = False
        self.encoder_conv_32[0].bias.requires_grad = False
        self.encoder_conv_40[0].weight.requires_grad = False
        self.encoder_conv_40[0].bias.requires_grad = False
        self.encoder_conv_41[0].weight.requires_grad = False
        self.encoder_conv_41[0].bias.requires_grad = False
        self.encoder_conv_42[0].weight.requires_grad = False
        self.encoder_conv_42[0].bias.requires_grad = False

    def freeze_seg(self):
        self.decoder_convtr_42[0].weight.requires_grad = False
        self.decoder_convtr_42[0].bias.requires_grad = False
        self.decoder_convtr_41[0].weight.requires_grad = False
        self.decoder_convtr_41[0].bias.requires_grad = False
        self.decoder_convtr_40[0].weight.requires_grad = False
        self.decoder_convtr_40[0].bias.requires_grad = False
        self.decoder_convtr_32[0].weight.requires_grad = False
        self.decoder_convtr_32[0].bias.requires_grad = False
        self.decoder_convtr_31[0].weight.requires_grad = False
        self.decoder_convtr_31[0].bias.requires_grad = False
        self.decoder_convtr_30[0].weight.requires_grad = False
        self.decoder_convtr_30[0].bias.requires_grad = False
        self.decoder_convtr_22[0].weight.requires_grad = False
        self.decoder_convtr_22[0].bias.requires_grad = False
        self.decoder_convtr_21[0].weight.requires_grad = False
        self.decoder_convtr_21[0].bias.requires_grad = False
        self.decoder_convtr_20[0].weight.requires_grad = False
        self.decoder_convtr_20[0].bias.requires_grad = False
        self.decoder_convtr_11[0].weight.requires_grad = False
        self.decoder_convtr_11[0].bias.requires_grad = False
        self.decoder_convtr_10[0].weight.requires_grad = False
        self.decoder_convtr_10[0].bias.requires_grad = False
        self.decoder_convtr_01[0].weight.requires_grad = False
        self.decoder_convtr_01[0].bias.requires_grad = False
        self.decoder_convtr_00[0].weight.requires_grad = False
        self.decoder_convtr_00[0].bias.requires_grad = False

    def init_vgg_weights(self):
        # assert self.encoder_conv_00[0].weight.size() == self.vgg16.features[0].weight.size()
        # self.encoder_conv_00[0].weight.data = self.vgg16.features[0].weight.data
        # assert self.encoder_conv_00[0].bias.size() == self.vgg16.features[0].bias.size()
        # self.encoder_conv_00[0].bias.data = self.vgg16.features[0].bias.data

        assert self.encoder_conv_01[0].weight.size() == self.vgg16.features[2].weight.size()
        self.encoder_conv_01[0].weight.data = self.vgg16.features[2].weight.data
        assert self.encoder_conv_01[0].bias.size() == self.vgg16.features[2].bias.size()
        self.encoder_conv_01[0].bias.data = self.vgg16.features[2].bias.data

        assert self.encoder_conv_10[0].weight.size() == self.vgg16.features[5].weight.size()
        self.encoder_conv_10[0].weight.data = self.vgg16.features[5].weight.data
        assert self.encoder_conv_10[0].bias.size() == self.vgg16.features[5].bias.size()
        self.encoder_conv_10[0].bias.data = self.vgg16.features[5].bias.data

        assert self.encoder_conv_11[0].weight.size() == self.vgg16.features[7].weight.size()
        self.encoder_conv_11[0].weight.data = self.vgg16.features[7].weight.data
        assert self.encoder_conv_11[0].bias.size() == self.vgg16.features[7].bias.size()
        self.encoder_conv_11[0].bias.data = self.vgg16.features[7].bias.data

        assert self.encoder_conv_20[0].weight.size() == self.vgg16.features[10].weight.size()
        self.encoder_conv_20[0].weight.data = self.vgg16.features[10].weight.data
        assert self.encoder_conv_20[0].bias.size() == self.vgg16.features[10].bias.size()
        self.encoder_conv_20[0].bias.data = self.vgg16.features[10].bias.data

        assert self.encoder_conv_21[0].weight.size() == self.vgg16.features[12].weight.size()
        self.encoder_conv_21[0].weight.data = self.vgg16.features[12].weight.data
        assert self.encoder_conv_21[0].bias.size() == self.vgg16.features[12].bias.size()
        self.encoder_conv_21[0].bias.data = self.vgg16.features[12].bias.data

        assert self.encoder_conv_22[0].weight.size() == self.vgg16.features[14].weight.size()
        self.encoder_conv_22[0].weight.data = self.vgg16.features[14].weight.data
        assert self.encoder_conv_22[0].bias.size() == self.vgg16.features[14].bias.size()
        self.encoder_conv_22[0].bias.data = self.vgg16.features[14].bias.data

        assert self.encoder_conv_30[0].weight.size() == self.vgg16.features[17].weight.size()
        self.encoder_conv_30[0].weight.data = self.vgg16.features[17].weight.data
        assert self.encoder_conv_30[0].bias.size() == self.vgg16.features[17].bias.size()
        self.encoder_conv_30[0].bias.data = self.vgg16.features[17].bias.data

        assert self.encoder_conv_31[0].weight.size() == self.vgg16.features[19].weight.size()
        self.encoder_conv_31[0].weight.data = self.vgg16.features[19].weight.data
        assert self.encoder_conv_31[0].bias.size() == self.vgg16.features[19].bias.size()
        self.encoder_conv_31[0].bias.data = self.vgg16.features[19].bias.data

        assert self.encoder_conv_32[0].weight.size() == self.vgg16.features[21].weight.size()
        self.encoder_conv_32[0].weight.data = self.vgg16.features[21].weight.data
        assert self.encoder_conv_32[0].bias.size() == self.vgg16.features[21].bias.size()
        self.encoder_conv_32[0].bias.data = self.vgg16.features[21].bias.data

        assert self.encoder_conv_40[0].weight.size() == self.vgg16.features[24].weight.size()
        self.encoder_conv_40[0].weight.data = self.vgg16.features[24].weight.data
        assert self.encoder_conv_40[0].bias.size() == self.vgg16.features[24].bias.size()
        self.encoder_conv_40[0].bias.data = self.vgg16.features[24].bias.data

        assert self.encoder_conv_41[0].weight.size() == self.vgg16.features[26].weight.size()
        self.encoder_conv_41[0].weight.data = self.vgg16.features[26].weight.data
        assert self.encoder_conv_41[0].bias.size() == self.vgg16.features[26].bias.size()
        self.encoder_conv_41[0].bias.data = self.vgg16.features[26].bias.data

        assert self.encoder_conv_42[0].weight.size() == self.vgg16.features[28].weight.size()
        self.encoder_conv_42[0].weight.data = self.vgg16.features[28].weight.data
        assert self.encoder_conv_42[0].bias.size() == self.vgg16.features[28].bias.size()
        self.encoder_conv_42[0].bias.data = self.vgg16.features[28].bias.data

    def load_segonly_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.replace('module.','')
            if name not in own_state:
                print ("[not exist]" + name)
                continue
            # print (name)
            own_state[name].copy_(param.data)