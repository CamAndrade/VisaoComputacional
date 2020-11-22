import logging
from typing import (overload)
from abc import abstractmethod, ABC

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Parâmetros treináveis: {}'.format(params))
        self.logger.info(self)

    @overload
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\n Parâmetros treináveis: {}'.format(params)

    @staticmethod
    def conv_block(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False):
        """
        A maioria das camadas convolucionais desta arquitetura, contem o kernel_size por padrão 3.
        :return: retorna camada convolucional
        """
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def triple_conv_block(self, in_channel, out_channel, padding=0):
        return nn.Sequential(
            self.conv_block(in_channel=in_channel, out_channel=out_channel),
            self.conv_block(in_channel=out_channel, out_channel=in_channel, kernel_size=1, padding=padding),
            self.conv_block(in_channel=in_channel, out_channel=out_channel)
        )

    @staticmethod
    def maxpooling():
        """
        Valores padrões da arquitetura
        :return: retorna camada maxpooling
        """
        return nn.MaxPool2d(kernel_size=2, stride=2)
