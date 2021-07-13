import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
from model.pa_model import model
from config import opt
from dataloader import get_loader
from utils import clip_gradient
from model.srm import srm
import numpy as np
import cv2
class trainer(object):
    def __init__(self, data, config):
        self.data = data
        self.lr = config.lr[1]*10
        self.max_epoch = config.epoch
        self.trainsize = config.trainsize
        self.modelpath = config.modelpath
        self.clip = config.clip
        self.build_model()
    def build_model(self):
        self.model = model(mode='srm').cuda()
        self.model.eval()
        self.model.load_state_dict(torch.load(self.modelpath))
        self.srm = srm().cuda()
        self.srm.train()
        self.optimizer = torch.optim.Adam(self.srm.parameters(), self.lr)
        self.loss_ce = torch.nn.BCELoss()
    def getImageVar(self, x):
        b, c, w, h = x.size()
        out4 = np.zeros((b, 4, 4))
        out2 = np.zeros((b, 2, 2))
        out1 = np.zeros((b, 1, 1))
        mean_img, std_img = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for i in range(b):
            ref_img = x[i, :, :, :].transpose(0, 1).transpose(1, 2)
            ref_img = ref_img.data.cpu().numpy()
            ref_img = (255*(ref_img*std_img+mean_img)).astype(np.uint8)
            ref = cv2.Laplacian(ref_img, cv2.CV_64F).var()
            for m in range(4):
                for n in range(4):
                    term = ref_img[m*64:(m+1)*64, n*64:(n+1)*64, :]
                    s = cv2.Laplacian(term, cv2.CV_64F).var()
                    if(s>=ref): out4[i, m, n] = 1
            for m in range(2):
                for n in range(2):
                    term = ref_img[m*128:(m+1)*128, n*128:(n+1)*128, :]
                    s = cv2.Laplacian(term, cv2.CV_64F).var()
                    if(s>=ref): out2[i, m, n] = 1
            for m in range(1):
                for n in range(1):
                    term = ref_img[m*256:(m+1)*256, n*256:(n+1)*256, :]
                    s = cv2.Laplacian(term, cv2.CV_64F).var()
                    if(s>=ref): out1[i, m, n] = 1
        out4 = torch.from_numpy(out4).cuda()
        out2 = torch.from_numpy(out2).cuda()
        out1 = torch.from_numpy(out1).cuda()
        return out4, out2, out1

    def train(self):
        total_step = len(self.data)
        for epoch in range(self.max_epoch):
            for i, pack in enumerate(self.data):
                images, gts, focal = pack
                focal = F.interpolate(focal, size=(self.trainsize, self.trainsize), mode='nearest')
                basize, dim, height, width = focal.size()
                images, gts, focal = images.cuda(), gts.cuda(), focal.cuda()
                images, gts, focal = Variable(images), Variable(gts), Variable(focal)
                focal = focal.view(1, basize, dim, height, width).transpose(0, 1)
                focal = torch.cat(torch.chunk(focal, 12, dim=2), dim=1)
                focal = torch.cat(torch.chunk(focal, 12, dim=1), dim=0).squeeze()
                with torch.no_grad():
                    out_rgb, x2r = self.model(images)
                    out_focal, x2f = self.model(focal)
                self.optimizer.zero_grad()
                out1, dis1_1, dis1_2 = self.srm(x2f, x2r, 1)
                out2, dis2_1, dis2_2 = self.srm(x2f, x2r, 2)
                out4, dis4_1, dis4_2 = self.srm(x2f, x2r, 4)
                gt4, gt2, gt1 = self.getImageVar(focal)
                gt4, gt2, gt1 = torch.tensor(gt4, dtype=torch.float32).cuda(), torch.tensor(gt2,dtype=torch.float32).cuda(), torch.tensor(gt1, dtype=torch.float32).cuda()
                loss1 = self.loss_ce(out1, gt1) #+ dis1_1 + dis1_2
                loss2 = self.loss_ce(out2, gt2) #+ dis2_1 + dis2_2
                loss4 = self.loss_ce(out4, gt4) #+ dis4_1 + dis4_2
                loss = loss1 # + loss2 + loss4
                loss.backward()
                clip_gradient(self.optimizer, self.clip)
                self.optimizer.step()
                if i % 10 == 0 or i == total_step:
                    print('epoch {:03d}, step {:04d}, loss1: {:.4f} loss2: {:0.4f} loss4: {:0.4f}'
                          .format(epoch, i,loss1.item(),loss2.item(), loss4.item()))
                save_path = 'ckpt/'
                if (i + 1) % 10 == 0:
                    torch.save(self.srm.state_dict(), save_path + 'srm.pth' + '.%d' % i)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if (epoch + 1) % 1 == 0:
                torch.save(self.srm.state_dict(), save_path + 'srm.pth' + '.%d' % epoch)
if __name__ == '__main__':
    config = opt
    train_loader = get_loader(config.img_root, config.gt_root, config.focal_root, batchsize=config.batchsize, trainsize=256)
    train = trainer(train_loader, config)
    train.train()

