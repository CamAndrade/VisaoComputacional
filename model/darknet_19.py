from .base_model import *
from .global_avg_pool2d import *


class Darknet(BaseModel):

    def __init__(self):
        super(Darknet, self).__init__()

        self.feature = nn.Sequential(
            self.conv_block(in_channel=3, out_channel=32),
            self.maxpooling(),
            self.conv_block(in_channel=32, out_channel=64),
            self.maxpooling(),
            self.triple_conv_block(in_channel=64, out_channel=128),
            self.maxpooling(),
            self.triple_conv_block(in_channel=128, out_channel=256),
            self.maxpooling(),
            self.quintuple_conv_block(in_channel=256, out_channel=512),
            self.maxpooling(),
            self.quintuple_conv_block(in_channel=512, out_channel=1024)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 1000, kernel_size=1),
            GlobalAvgPool2d(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

    def quintuple_conv_block(self, in_channel, out_channel):
        return nn.Sequential(
            self.conv_block(in_channel=in_channel, out_channel=out_channel),
            self.conv_block(in_channel=out_channel, out_channel=in_channel, kernel_size=1, padding=0),
            self.conv_block(in_channel=in_channel, out_channel=out_channel),
            self.conv_block(in_channel=out_channel, out_channel=in_channel, kernel_size=1, padding=0),
            self.conv_block(in_channel=in_channel, out_channel=out_channel)
        )

    def __del__(self):
        pass
