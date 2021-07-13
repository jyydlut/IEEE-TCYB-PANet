import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
from dataloader import get_loader
from config import opt
from utils import clip_gradient, adjust_lr
from model.pa_model import model, decoder
from model.mslm import mslm
from model.srm import srm
from func import fused, fusedidx

class trainer(object):
    def __init__(self, data, config):
        self.data = data
        self.lr = config.lr[0]
        self.max_epoch = config.epoch
        self.triansize = config.trainsize
        self.decay_rate = config.decay_rate
        self.decay_epoch = config.decay_epoch
        self.modelpath = config.modelpath
        self.mslmpath = config.mslmpath
        self.srmpath = config.srmpath
        self.clip = config.clip
        self.build_model()
    def build_model(self):
        self.model = model(mode='second_decoder').cuda()
        self.model.eval()
        self.model.load_state_dict(torch.load(self.modelpath))
        self.mslm = mslm().cuda()
        self.mslm.eval()
        self.mslm.load_state_dict(torch.load(self.mslmpath))
        self.srm = srm().cuda()
        self.srm.eval()
        self.srm.load_state_dict(torch.load(self.srmpath))
        self.decoder = decoder(channels=32).cuda()
        self.decoder.eval()
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), self.lr)
        self.loss_bce = torch.nn.BCELoss()
    def train(self):
        for epoch in range(self.max_epoch):
            for i, pack in enumerate(self.data):
                total_step = len(self.data)
                images, gts, focal = pack
                focal = F.interpolate(focal, size=(self.triansize, self.triansize), mode='nearest')
                basize, dim, height, width = focal.size()
                images, gts, focal = images.cuda(), gts.cuda(), focal.cuda()
                images, gts, focal = Variable(images), Variable(gts), Variable(focal)
                focal = focal.view(1, basize, dim, height, width).transpose(0, 1)
                focal = torch.cat(torch.chunk(focal, 12, dim=2), dim=1)
                focal = torch.cat(torch.chunk(focal, basize, dim=0), dim=1).squeeze()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    out_rgb, xr, x2r, featrgb = self.model(images)
                    out_focal, xf, x2f, featfocal = self.model(focal)
                    x5focal, x4focal, x3focal = featfocal
                    x5out = fused(x5focal, 1, fusedidx(xf, x2f, xr, x2r, self.mslm, self.srm, 1)).cuda()
                    x4out = fused(x4focal, 2, fusedidx(xf, x2f, xr, x2r, self.mslm, self.srm, 2)).cuda()
                    x3out = fused(x3focal, 4, fusedidx(xf, x2f, xr, x2r, self.mslm, self.srm, 4)).cuda()
                out, _ = self.decoder(x5out, x4out, x3out)
                out = torch.sigmoid(out)
                loss = self.loss_bce(out, gts)
                loss.backward()
                clip_gradient(self.optimizer, self.clip)
                self.optimizer.step()
                if i % 10 == 0 or i == total_step:
                    print('epoch {:03d}, step {:04d}, loss: {:.4f}'.format(epoch, i, loss.item()))
            adjust_lr(self.optimizer, self.lr, epoch, self.decay_rate, self.decay_epoch)
            save_path = 'ckpt/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if (epoch + 1) % 1 == 0:
                torch.save(self.decoder.state_dict(), save_path + 'sec_de.pth' + '.%d' % epoch)
if __name__ == '__main__':
    config = opt
    train_loader = get_loader(opt.img_root, opt.gt_root, opt.focal_root, batchsize=8, trainsize=opt.trainsize)
    train = trainer(train_loader, config)
    train.train()

