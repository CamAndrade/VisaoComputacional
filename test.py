import torchvision.models as models
from model.darknet_19 import Darknet
from model.dark_covid_net import DarkCovidNet
import torchsummary

resnet18 = models.resnet18(pretrained=True)
model = Darknet()
# model = DarkCovidNet()

torchsummary.summary(model, (3,256,256), batch_size=32)