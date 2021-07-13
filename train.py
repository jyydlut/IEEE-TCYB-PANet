import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
from dataloader import get_loader
from utils import clip_gradient, adjust_lr, loss_position, loss_boundary
import torch.optim
from model.pa_model import model
from config import opt
class trainer(object):
    def __init__(self, data, config):
        self.data = data
        self.lr = config.lr[0]
        self.max_epoch = config.epoch
        self.triansize = config.trainsize
        self.decay_rate = config.decay_rate
        self.decay_epoch = config.decay_epoch
        self.clip = config.clip
        self.build_model()
    def build_model(self):
        self.model = model(mode='sal').cuda()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.loss_bce = torch.nn.BCELoss()
        self.loss_p = loss_position
        self.loss_b = loss_boundary
    def train(self):
        for epoch in range(self.max_epoch):
            for i, pack in enumerate(self.data):
                total_step = len(self.data)
                images, gts, focal = pack
                focal = F.interpolate(focal, size=(self.triansize , self.triansize ), mode='nearest')
                basize, dim, height, width = focal.size()
                images, gts, focal = images.cuda(), gts.cuda(), focal.cuda()
                images, gts, focal = Variable(images), Variable(gts), Variable(focal)
                gts_focal = gts.repeat(1, 12, 1, 1)
                focal = focal.view(1, basize, dim, height, width).transpose(0, 1)
                gts_focal = gts_focal.view(1, basize, 12, height, width).transpose(0, 1)
                focal = torch.cat(torch.chunk(focal, 12, dim=2), dim=1)
                gts_focal = torch.cat(torch.chunk(gts_focal, 12, dim=2), dim=1)
                focal = torch.cat(torch.chunk(focal, basize, dim=0), dim=1).squeeze()
                gts_focal = torch.cat(torch.chunk(gts_focal, basize, dim=0), dim=1).squeeze(0)
                self.optimizer.zero_grad()
                out_rgb = self.model(images)
                loss_c = self.loss_p(2, out_rgb, gts) + self.loss_p(1, out_rgb, gts) + self.loss_p(0, out_rgb, gts)
                loss_b = self.loss_b(out_rgb, gts)
                loss_r = self.loss_bce(out_rgb, gts) + loss_b + 0.05*loss_c
                out_focal = self.model(focal)
                loss_f = self.loss_bce(out_focal, gts_focal)
                loss = loss_r+loss_f
                loss.backward()
                clip_gradient(self.optimizer, self.clip)
                self.optimizer.step()
                if i % 10 == 0 or i == total_step:
                    print('epoch {:03d}, step {:04d}, lossr: {:.4f} lossf: {:0.4f}'. format(epoch, i, loss_r.item(), loss_f.item()))
            adjust_lr(self.optimizer, self.lr, epoch, self.decay_rate, self.decay_epoch)
            save_path = 'ckpt/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if (epoch + 1) % 1 == 0:
                torch.save(self.model.state_dict(), save_path + 'model.pth' + '.%d' % epoch)
if __name__ == '__main__':
    config = opt
    train_loader = get_loader(opt.img_root, opt.gt_root, opt.focal_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train = trainer(train_loader, config)
    train.train()