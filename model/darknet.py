import torch.nn as nn
from .base_model import *


class Darknet(BaseModel):

    def __init__(self):
        super(Darknet, self).__init__()

        self.feature = nn.Sequential(
            self.conv_block(in_channel=3, out_channel=32),
            self.maxpooling(),
            self.conv_block(in_channel=32, out_channel=64),
            self.maxpooling(),
            self.triple_conv_block(in_channel=64, out_channel=128),
            self.maxpooling(),
            self.triple_conv_block(in_channel=128, out_channel=256),
            self.maxpooling(),
            self.quintuple_conv_block(in_channel=256, out_channel=512),
            self.maxpooling(),
            self.quintuple_conv_block(in_channel=512, out_channel=1024)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1000, kernel_size=1),
            #avgpoll global
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def conv_block(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False):
        """
        A maioria das camadas convolucionais desta arquitetura, contem o kernel_size por padrão 3.
        :param in_channel:
        :param out_channel:
        :param kernel_size:
        :param padding:
        :param stride:
        :param bias:
        :return: retorna camada convolucional
        """
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def triple_conv_block(self, in_channel, out_channel):
        return nn.Sequential(
            self.conv_block(in_channel=in_channel, out_channel=out_channel),
            self.conv_block(in_channel=out_channel, out_channel=in_channel, kernel_size=1, padding=0),
            self.conv_block(in_channel=in_channel, out_channel=out_channel)
        )

    def quintuple_conv_block(self, in_channel, out_channel):
        return nn.Sequential(
            self.conv_block(in_channel=in_channel, out_channel=out_channel),
            self.conv_block(in_channel=out_channel, out_channel=in_channel, kernel_size=1, padding=0),
            self.conv_block(in_channel=in_channel, out_channel=out_channel),
            self.conv_block(in_channel=out_channel, out_channel=in_channel, kernel_size=1, padding=0),
            self.conv_block(in_channel=in_channel, out_channel=out_channel)
        )

    @staticmethod
    def maxpooling():
        """
        Valores padrões da arquitetura
        :return: retorna camada maxpooling
        """
        return nn.MaxPool2d(kernel_size=2, stride=2)

    def __del__(self):
        pass