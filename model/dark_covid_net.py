from .base_model import *


class DarkCovidNet(BaseModel):

    def __init__(self):
        super(DarkCovidNet, self).__init__()

        self.feature = nn.Sequential(
            self.conv_block(in_channel=3, out_channel=8),
            self.maxpooling(),
            self.conv_block(in_channel=8, out_channel=16),
            self.maxpooling(),
            self.triple_conv_block(in_channel=16, out_channel=32, padding=1),
            self.maxpooling(),
            self.triple_conv_block(in_channel=32, out_channel=64, padding=1),
            self.maxpooling(),
            self.triple_conv_block(in_channel=64, out_channel=128, padding=1),
            self.maxpooling(),
            self.triple_conv_block(in_channel=128, out_channel=256, padding=1),
            self.conv_block(in_channel=256, out_channel=128, kernel_size=1),
            self.conv_block(in_channel=128, out_channel=256)
        )

        self.classifier = nn.Sequential(
            self.conv_layer(in_channel=256, out_channel=3),
            nn.Flatten(),
            nn.Linear(507, 3)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def conv_layer(in_channel, out_channel, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )