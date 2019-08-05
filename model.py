import torch.nn as nn
import torch.nn.functional as F

class SiamFc(nn.Module):
    def __init__(self):
        super(SiamFc, self).__init__()
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 192, 3, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(192, 192, 3, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(192, 128, 3, 1),
            nn.BatchNorm2d(128))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out',
                                        nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def exemplar(self,x):
        self.template=self.feature(x)
    def forward(self,x):
        feature=self.feature(x)
        n,c,h,w=feature.size()

        feature=feature.view(1,n*c,h,w)
        response_map=F.conv2d(feature,self.template,groups=n)
        response_map.view(n,1,response_map.size(-2),response_map.size(-1))
        response_map=response_map*0.001
        return response_map

    
