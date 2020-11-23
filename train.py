import numpy as np
from utils import Dataset
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import ModelSelector, DeviceSelector
import logging

"""
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


class Train:
    def __init__(self, dataset_object, device, model, epochs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__lacc = [0]
        self.__lloss = [0]
        self.__epochs = epochs
        self.__train_model(dataset_object, device, model)

    def __train(self, dataloader_train, model, optimizer, criterion, device):
        model.to(device)
        model.train()
        self.lloss.clear()
        self.logger.info("Epochs : %4.3f" %self.epochs)
        for epoch in tqdm(range(self.epochs)):
            for x, y in dataloader_train:
                x = x.to(device)
                y = y.to(device)

                prediction = model(x)
                loss = criterion(prediction, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.lloss.append(loss.item())

            self.logger.info("Loss treino: %4.3f" % (np.mean(self.lloss)))

    def __evaluate(self, dataloader_test, model, criterion, device):
        model.to(device)
        model.eval()

        self.lloss.clear()
        self.lacc.clear()

        with torch.no_grad():
            # correct = 0
            # total = 0
            for x, y in tqdm(dataloader_test):
                x = x.to(device)
                prediction = model(x)

                loss = criterion(prediction, y.to(device))
                y_pred = prediction.argmax(dim=1)

                acc = accuracy_score(y, y_pred)
                self.lacc.append(acc)
                self.lloss.append(loss.item())


                # outputs = model(x)
                # _, predicted = torch.max(outputs.data, 1)
                # total += y.size(0)
                # correct += (predicted == y).sum().item()

        return np.mean(self.lacc), np.mean(self.lloss)

    def __train_model(self, ds_object, device, model, learning_rate=3e-3):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for train, test in ds_object.skf.split(range(len(ds_object.dataset)), ds_object.dataset.targets):
            ds_train, ds_test = ds_object.treino_teste(ds_object.dataset, (train, test))

            dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=32)
            dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=32)

            self.logger.info("------ Treino ------ ")
            self.__train(dataloader_train, model, optimizer, criterion, device)

            self.logger.info("------ Avaliação ------")
            (acc, loss) = self.__evaluate(dataloader_test, model, criterion, device)

            self.logger.info("Accuracy:%4.3f Loss:%4.3f" % (acc, loss))

            torch.save(model.state_dict(), 'model.pth')
            self.lacc.append(acc)

        np.mean(self.lacc), np.std(self.lacc)

    @property
    def lacc(self):
        return self.__lacc

    @lacc.setter
    def lacc(self, value):
        self.__lacc = value

    @property
    def lloss(self):
        return self.__lloss

    @lloss.setter
    def lloss(self, value):
        self.__lloss = value

    @property
    def epochs(self):
        return int(self.__epochs)

    @epochs.setter
    def epochs(self, value):
        self.__epochs = value


if __name__ == '__main__':
    model = ModelSelector()
    device = DeviceSelector()
    dataset_object = Dataset(model.imagesDir)

    Train(dataset_object, device.device, model.model, model.epochs)
