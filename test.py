import torchvision.models as models
from model.darknet import Darknet
import torchsummary

resnet18 = models.resnet18(pretrained=True)
model = Darknet()

torchsummary.summary(model, (3,256,256), batch_size=32)