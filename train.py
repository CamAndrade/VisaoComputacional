import argparse
import numpy as np
from utils import Dataset
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from utils import ModelSelector, DeviceSelector
from metrics import Metrics
import os
from pytorch_lightning import loggers as pl_loggers
import json

"""
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Realiza o treinamento da CNN de acordo com os parâmetros informados.')
    parser.add_argument('--imagesDir', '-id', required=True, help='Diretório raiz do dataset.')
    parser.add_argument('--model', '-m', required=True, default=False, choices=['darknet19', 'darkCovidNet'],
                        help='Modelos.')
    parser.add_argument('--epochs', '-e', required=True, help='Quantidade de épocas para treino.')
    parser.add_argument('--pth_path', '-pth', required=True, help='Diretório para salvar os modelos treinados.')
    parser.add_argument('--log_path', '-log', required=True, help='Diretório para salvar os logs.')
    return parser.parse_args()


class Train:
    def __init__(self, dataset_object, device, model, epochs, pth_path, log_path):
        self.__classes = dataset_object.classes
        self.__epochs = epochs
        self.__fold = 0
        self.__model_name = model
        self.__train_model(ds_object=dataset_object, device=device, model=model, learning_rate=3e-3, pth_path=pth_path, log_path=log_path)

    def __train(self, dataloader_train, dataloader_test, model, optimizer, criterion, device, log_path):
        logger = pl_loggers.TensorBoardLogger(log_path, name=None, version=None)

        for epoch in range(self.epochs):
            model.train()

            train_acc = list()
            train_loss = list()
            for x, y in dataloader_train:
                x = self.__to_norm(x)

                x = x.to(device)
                y = y.to(device)

                prediction = model(x)
                loss = criterion(prediction, y)
                y_pred = prediction.cpu().argmax(dim=1)
                acc = accuracy_score(y.cpu(), y_pred)

                train_acc.append(acc)
                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # calcula a acurácia do conjunto de teste a cada época
            val_acc, val_loss = self.__evaluate(dataloader_test, model, criterion, device)

            logger.experiment.add_scalar('train_acc', np.mean(train_acc), epoch)
            logger.experiment.add_scalar('train_loss', np.mean(train_loss), epoch)
            logger.experiment.add_scalar('val_acc', val_acc, epoch)
            logger.experiment.add_scalar('val_loss', val_loss, epoch)
            print(
                'model:', self.model_name,
                'fold:', self.fold,
                'epoch:', epoch,
                'train_acc:', np.mean(train_acc),
                'train_loss:', np.mean(train_loss),
                'val_acc:', val_acc,
                'val_loss', val_loss
            )

    def __evaluate(self, dataloader_test, model, criterion, device):
        val_acc = list()
        val_loss = list()

        model.eval()
        with torch.no_grad():
            for x, y in dataloader_test:
                x = self.__to_norm(x)

                x = x.to(device)

                prediction = model(x)
                loss = criterion(prediction, y.to(device))
                y_pred = prediction.cpu().argmax(dim=1)
                acc = accuracy_score(y, y_pred)

                val_acc.append(acc)
                val_loss.append(loss.item())
        return np.mean(val_acc), np.mean(val_loss)

    @staticmethod
    def __to_norm(imagens):
        """Normalização por batch"""
        media = torch.mean(imagens)
        desvio_padrao = torch.std(imagens)
        return (imagens - media) / desvio_padrao

    def __update_darknet19_output(self, model_):
        model_.classifier[0] = nn.Conv2d(1024, len(self.__classes), kernel_size=1)
        return model_

    def __update_darkCovidNet_output(self, model_):
        if len(self.__classes) == 2:
            model_.classifier[0] = model_.conv_layer(256, 2)
            model_.classifier[2] = nn.Linear(338, 2)
        if len(self.__classes) == 3:
            model_.classifier[0] = model_.conv_layer(256, 3)
            model_.classifier[2] = nn.Linear(507, 3)
        return model_

    switcher = {
        'darknet19': __update_darknet19_output,
        'darkCovidNet': __update_darkCovidNet_output
    }

    def __train_model(self, ds_object, device, model, learning_rate=3e-3, pth_path ='', log_path =''):
        for train, test in ds_object.skf.split(range(len(ds_object.dataset)), ds_object.dataset.targets):
            # train e test são os índices utilizados em cada fold para treino e teste

            model_ = ModelSelector(model).model
            criterion = nn.CrossEntropyLoss()

            # altera a camada de classificação do modelo de acordo com a quantidade de classes
            get_model = self.switcher.get(model)
            model_ = get_model(self, model_)

            if self.fold == 0:
                print(model_)

            optimizer = optim.Adam(model_.parameters(), lr=learning_rate)

            ds_train, ds_test = ds_object.treino_teste(ds_object.dataset, (train, test))
            dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=32)
            dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=1)

            pth_fold_path = os.path.join(pth_path, model + '/fold_' + str(self.__fold))
            log_fold_path = os.path.join(log_path, model + '/fold_' + str(self.__fold))
            if not os.path.exists(pth_fold_path):
                os.makedirs(pth_fold_path)

            model_.to(device)
            self.__train(dataloader_train, dataloader_test, model_, optimizer, criterion, device, log_fold_path)

            # calcula as métricas para a fold atual após o treinamento
            y_true = list()
            y_pred = list()
            model_.eval()
            with torch.no_grad():
                for x, y in dataloader_test:
                    x = self.__to_norm(x)
                    x = x.to(device)
                    prediction = model_(x)
                    pred = prediction.cpu().argmax(dim=1)
                    y_true.append(y.item())
                    y_pred.append(pred.item())

            model_save_path = os.path.join(pth_fold_path, 'model.pth')
            torch.save(model_.state_dict(), model_save_path)

            self.__salvar_metricas(pth_fold_path, y_true, y_pred)
            self.fold += 1

    def __salvar_metricas(self, path, y_true, y_pred):
        metricas_path = os.path.join(path, 'metricas.json')
        metricas = Metrics(y_true, y_pred)
        metricas_json = dict(
            classes=self.__classes,
            matriz_confusao=metricas.matriz_confusao().tolist(),
            sensibilidade=metricas.sensibilidade().item(),
            especificidade=metricas.especificidade().item(),
            precisao=metricas.precisao().item(),
            medida_f=metricas.medida_f().item(),
            acuracia=metricas.acuracia().item()
        )
        with open(metricas_path, 'w') as json_path:
            json.dump(metricas_json, json_path)
        print(metricas_json)
        print()

    @property
    def fold(self):
        return self.__fold

    @fold.setter
    def fold(self, value):
        self.__fold = value

    @property
    def model_name(self):
        return self.__model_name

    @model_name.setter
    def model_name(self, value):
        self.__model_name = value

    @property
    def epochs(self):
        return int(self.__epochs)

    @epochs.setter
    def epochs(self, value):
        self.__epochs = value


if __name__ == '__main__':
    args = parse_args()
    device = DeviceSelector()
    dataset_object = Dataset(args.imagesDir)
    Train(dataset_object, device.device, args.model, args.epochs, args.pth_path, args.log_path)

    # para realizar todos os experimentos com a mesma seed
    # é necessário atualizar a seed do dataset
    # seed = 3
    # device = DeviceSelector()
    #
    # dataset_object = Dataset('dataset2')
    # Train(dataset_object, device.device, 'darknet19', 100, 'results2/' + str(seed), 'logs2/' + str(seed))
    # Train(dataset_object, device.device, 'darkCovidNet', 100, 'results2/' + str(seed), 'logs2/' + str(seed))
    #
    # dataset_object = Dataset('dataset3')
    # Train(dataset_object, device.device, 'darknet19', 100, 'results3/' + str(seed), 'logs3/' + str(seed))
    # Train(dataset_object, device.device, 'darkCovidNet', 100, 'results3/' + str(seed), 'logs3/' + str(seed))
