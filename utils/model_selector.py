from model import DarkCovidNet
from model import Darknet
import argparse


class ModelSelector:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--imagesDir', '-id', required=True, help='Diretório raiz do dataset.')
        parser.add_argument('--model', '-m', required=True, default=False, choices=['darknet19', 'darkCovidNet'],
                            help='Modelos.')
        parser.add_argument('--epochs', '-e', required=True, help='Quantidade de épocas para treino.')

        self.__args = parser.parse_args()
        self.__imagesDir = self.__args.imagesDir

        self.__epochs = self.__args.epochs

        metodo_model = self.switcher.get(self.__args.model, lambda: "Selecione um modelo válido.")
        self.__model = metodo_model(self)

        self.__show_model()

    def __darknet19(self):
        return Darknet()

    def __darkCovidNet(self):
        return DarkCovidNet()

    def __show_model(self):
        print("-------- Modelo --------")
        print(self.model)

    switcher = {
        "darknet19": __darknet19,
        "darkCovidNet": __darkCovidNet
    }

    @property
    def imagesDir(self):
        return self.__imagesDir

    @imagesDir.setter
    def imagesDir(self, value):
        self.__imagesDir = value

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, value):
        self.__epochs = value
