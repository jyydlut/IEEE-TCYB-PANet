import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
from config import opt
from dataloader import get_loader
from utils import clip_gradient
from model.pa_model import model
from model.mslm import mslm
from func import sal_gt, boundary_gt, pos_gt
class trainer(object):
    def __init__(self, data, config):
        self.data = data
        self.lr = config.lr[1]
        self.max_epoch = config.epoch
        self.trainsize = config.trainsize
        self.modelpath = config.modelpath
        self.clip = config.clip
        self.build_model()
    def build_model(self):
        self.model = model(mode='mslm').cuda()
        self.model.eval()
        self.model.load_state_dict(torch.load(self.modelpath))
        self.mslm = mslm().cuda()
        self.mslm.train()
        self.optimizer = torch.optim.Adam(self.mslm.parameters(), self.lr)
        self.loss_ce = torch.nn.BCELoss()
    def cal_gt(self, in1, in2, num):
        basize = in2.size()[0]
        gt1, gt2, gt3 = sal_gt(in1, in2, num).view(-1, 1).cuda(), pos_gt(in1, in2, num).view(-1, 1).cuda(), boundary_gt(in1, in2, num).view(-1, 1).cuda()
        gt1 = torch.zeros(basize * num * num, 12).cuda().scatter_(1, gt1, 1).view(basize * num * num, 1, 12)
        gt2 = torch.zeros(basize * num * num, 12).cuda().scatter_(1, gt2, 1).view(basize * num * num, 1, 12)
        gt3 = torch.zeros(basize * num * num, 12).cuda().scatter_(1, gt3, 1).view(basize * num * num, 1, 12)
        gt = gt1 + gt2 + gt3
        gt[gt > 1] = 1
        gt = torch.cat(torch.chunk(gt, num * num, dim=0), dim=1)
        gt = torch.cat(torch.chunk(gt, 12, dim=2), dim=0)
        gt = torch.cat(torch.chunk(gt, num, dim=1), dim=2)
        return gt
    def train(self):
        total_step = len(self.data)
        for epoch in range(self.max_epoch):
            for i, pack in enumerate(self.data):
                images, gts, focal = pack
                focal = F.interpolate(focal, size=(self.trainsize , self.trainsize), mode='nearest')
                basize, dim, height, width = focal.size()
                images, gts, focal = images.cuda(), gts.cuda(), focal.cuda()
                images, gts, focal = Variable(images), Variable(gts), Variable(focal)
                focal = focal.view(1, basize, dim, height, width).transpose(0, 1)
                focal = torch.cat(torch.chunk(focal, 12, dim=2), dim=1)
                focal = torch.cat(torch.chunk(focal, 12, dim=1), dim=0).squeeze()
                with torch.no_grad():
                    out_rgb, xr = self.model(images)
                    out_focal, xf = self.model(focal)
                self.optimizer.zero_grad()
                out1 = self.mslm(xf, xr, 1)
                gt1 = self.cal_gt(out_focal, out_rgb, 1)
                out2 = self.mslm(xf, xr, 2)
                gt2 = self.cal_gt(out_focal, out_rgb, 2)
                out4 = self.mslm(xf, xr, 4)
                gt4 = self.cal_gt(out_focal, out_rgb, 4)
                loss1 = self.loss_ce(out1, gt1)
                loss2 = self.loss_ce(out2, gt2)
                loss4 = self.loss_ce(out4, gt4)
                loss = (loss1 + loss2 + loss4) / 3
                loss.backward()
                clip_gradient(self.optimizer, self.clip)
                self.optimizer.step()
                if i % 10 == 0 or i == total_step:
                    print('epoch {:03d}, step {:04d}, loss1: {:.4f} loss2: {:0.4f} loss4: {:0.4f}'
                          . format(epoch, i, loss1.item(), loss2.item(), loss4.item()))
            save_path = 'ckpt/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if (epoch + 1) % 1 == 0:
                torch.save(self.mslm.state_dict(), save_path + 'mslm.pth' + '.%d' % epoch)
if __name__ == '__main__':
    config = opt
    train_loader = get_loader(config.img_root, config.gt_root, config.focal_root, batchsize=config.batchsize, trainsize=256)
    train = trainer(train_loader, config)
    train.train()
