import torchvision.models as models
from model.darknet_19 import Darknet
from model.dark_covid_net import DarkCovidNet
from utils import Dataset
import torchvision
import torchsummary
import torch

resnet18 = models.resnet18(pretrained=True)
# print(resnet18)
# model = Darknet()
# model = DarkCovidNet()

# print(model)

a = Dataset("../dataset")
dataset = a.dataset

# a.show(dataset)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

print(len(dataset))
for train, test in a.skf.split(range(len(dataset)), dataset.targets):
    treino, teste = a.treino_teste(dataset, (train, test))

    print(len(treino))
    print(len(teste))

# torchsummary.summary(model, (3,256,256), batch_size=32)