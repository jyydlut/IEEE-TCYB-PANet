import torch
import torch.nn as nn
import torch.nn.functional as F
class mslm(nn.Module):
    def __init__(self):
        super(mslm, self).__init__()
        self.v = vggx()
    def forward(self, xf, xr, num):
        ba = xr.size()[0]
        inx = xf
        iny = xr.repeat(12, 1, 1, 1)
        inxy = torch.cat((inx, iny), dim=1)
        inxy = F.interpolate(inxy, size=(16*num, 16*num), mode='bilinear')
        out = torch.zeros(ba*12, num, num).cuda()
        for i in range(num):
            for j in range(num):
                out[:, i, j] = self.v(inxy[:, :, i*16:(i+1)*16, j*16:(j+1)*16]).squeeze()
        return out
class vggx(nn.Module):
    def __init__(self):
        super(vggx, self).__init__()
        self.bn1_1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.bn1_2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(64, 32, 3, 1, 1))
        conv1.add_module('bn1_1', self.bn1_1)
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv1.add_module('conv1_2', nn.Conv2d(32, 16, 3, 1, 1))
        conv1.add_module('bn1_2', self.bn1_2)
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.conv1 = conv1
        self.liner1 = nn.Linear(1024, 256)
        self.liner2 = nn.Linear(256, 1)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        ba = x.size()[0]
        x = self.conv1(x)
        x = x.view(ba, -1)
        x = self.liner1(x)
        x = self.liner2(x)
        x = torch.sigmoid(x)
        return x