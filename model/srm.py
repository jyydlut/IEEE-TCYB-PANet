import torch
import torch.nn as nn
import torch.nn.functional as F
class srm(nn.Module):
    def __init__(self):
        super(srm, self).__init__()
        self.v = vggx()
    def forward(self, xf, xr, num):
        ba, ch = xr.size()[0], xr.size()[1]
        inx = xf
        iny = xr.repeat(12, 1, 1, 1)
        inxy = torch.cat((inx, iny), dim=1)
        inxy = F.interpolate(inxy, size=(16*num, 16*num), mode='bilinear')
        out = torch.zeros(ba*12, num, num).cuda()
        for i in range(num):
            for j in range(num):
                out[:, i, j] = self.v(inxy[:, :, i*16:(i+1)*16, j*16:(j+1)*16]).squeeze()
        outc = out.view(ba*12, 1, num, num)
        outc = torch.cat(outc.chunk(12, dim=0), dim=1)
        loss1, loss2 = 0, 0
        for i in range(num):
            for j in range(num):
                feat = xf[:, :, i*16:(i+1)*16, j*16:(j+1)*16].view(ba*12, 1, ch, 16, 16)
                feat = torch.cat(feat.chunk(12, dim=0), dim=1)
                ck = outc[:, :, i:i+1, j:j+1].view(ba, 12, 1, 1, 1).repeat(1, 1, 128, 16, 16)
                t1 = torch.sum(ck*feat, dim=1, keepdim=True)/torch.sum(ck, dim=1, keepdim=True)
                t0 = torch.sum((1-ck)*feat, dim=1, keepdim=True)/torch.sum((1-ck), dim=1, keepdim=True)
                dis1_t1, dis1_t0 = t1.view(-1), t0.view(-1)
                loss1 = loss1 + torch.cosine_similarity(dis1_t1, dis1_t0, dim=0)
                t1 = t1.repeat(1, 12, 1, 1, 1)
                t0 = t0.repeat(1, 12, 1, 1, 1)
                dis0_t1, dis0_f1 = t1.view(-1), (ck*feat).view(-1)
                dis0_t0, dis0_f0 = t0.view(-1), ((1-ck)*feat).view(-1)
                loss2 = loss2-torch.cosine_similarity(dis0_t1, dis0_f1, dim=0)-torch.cosine_similarity(dis0_t0, dis0_f0, dim=0)+2
        loss1, loss2 = 1-loss1/(num*num), 1-loss2/(num*num)
        return out, loss1, loss2
class vggx(nn.Module):
    def __init__(self):
        super(vggx, self).__init__()
        self.bn1_1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.bn1_2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(256, 32, 3, 1, 1))
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