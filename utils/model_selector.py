from model import DarkCovidNet
from model import Darknet


class ModelSelector:
    def __init__(self, model):
        metodo_model = self.switcher.get(model, lambda: "Selecione um modelo válido.")
        self.__model = metodo_model(self)

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
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value
