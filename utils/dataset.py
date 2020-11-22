from PIL import Image
import torch
import torchvision

from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Dataset():
    def __init__(self, dir):
        self.__dir = dir
        self.__dataset = self.carregar_dataset()
        self.__skf = self.estratificacao()

    @staticmethod
    def __carregar_imagem():
        return lambda img: Image.open(img).convert('RGB')

    @staticmethod
    def __transformacao():
        return transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             ToNorm()]
        )

    def carregar_dataset(self):
        return torchvision.datasets.DatasetFolder(self.__dir,
                                                  loader=self.__carregar_imagem(),
                                                  extensions=('jpg', 'png', 'jpeg'),
                                                  transform=self.__transformacao())

    @staticmethod
    def treino_teste(dataset, listas):
        return [torch.utils.data.Subset(dataset, llist) for llist in listas]

    @staticmethod
    def estratificacao():
        """
        https://xzz201920.medium.com/stratifiedkfold-v-s-kfold-v-s-stratifiedshufflesplit-ffcae5bfdf
        :return: O artigo indica a avaliação com 5 folds para ambos os problemas (classificação binária e tripla)
        """
        return StratifiedKFold(n_splits=5)

    @staticmethod
    def show(dataset, batch_size=32, nrow=5):
        data_loader = DataLoader(dataset, batch_size=batch_size)
        data, class_att = next(iter(data_loader))
        grid_img = torchvision.utils.make_grid(data, nrow=nrow)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, value):
        self.__dataset = value

    @property
    def skf(self):
        return self.__skf

    @skf.setter
    def skf(self, value):
        self.__skf = value


class ToNorm(object):
    def __call__(self, imagem):
        media = torch.mean(imagem)
        desvio_padrao = torch.std(imagem)
        return (imagem - media) / desvio_padrao
