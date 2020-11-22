import torch


class DeviceSelector:
    def __init__(self):
        self.__device = self.__carrega_device()

    @staticmethod
    def __carrega_device():
        return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
