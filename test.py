import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy import misc
from dataloader import test_dataset
from model.pa_model import model, decoder
from func import fused, fusedidx
from model.mslm import mslm
from model.srm import srm
from config import opt
config=opt
model = model(mode='test')
model.load_state_dict(torch.load(config.modelpath))
model.cuda()
model.eval()
mslm = mslm()
mslm.load_state_dict(torch.load(config.mslmpath))
mslm.cuda()
mslm.eval()
srm = srm().cuda()
srm.load_state_dict(torch.load(config.srmpath))
srm.eval()
secd = decoder(32).cuda()
secd.load_state_dict(torch.load(config.secdecoder))
secd.eval()
test_datasets = ['DUTS', 'HFUT', 'LFSD']

save_path = './results/' + test_datasets[0] + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
test_loader = test_dataset(config.test_image, config.test_gt, config.test_focal, config.testsize)
for i in range(test_loader.size):
    image, focal, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    dim, height, width = focal.size()
    basize = 1
    focal = focal.view(1, basize, dim, height, width).transpose(0, 1)
    focal = torch.cat(torch.chunk(focal, 12, dim=2), dim=1)
    focal = torch.cat(torch.chunk(focal, 12, dim=1), dim=0).squeeze()
    focal = F.interpolate(focal, size=(256, 256), mode='bilinear')
    image, focal = image.cuda(), focal.cuda()
    with torch.no_grad():
        rgb_out, xr, x2r, featrgb = model(image)
        focal_out, xf, x2f, featfocal = model(focal)
        x5focal, x4focal, x3focal = featfocal
        x5out = fused(x5focal, 1, fusedidx(xf, x2f, xr, x2r, mslm, srm, 1)).cuda()
        x4out = fused(x4focal, 2, fusedidx(xf, x2f, xr, x2r, mslm, srm, 2)).cuda()
        x3out = fused(x3focal, 4, fusedidx(xf, x2f, xr, x2r, mslm, srm, 4)).cuda()
        out, _ = secd(x5out, x4out, x3out)
        out = torch.sigmoid(out)
    res = F.upsample(out, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    misc.imsave(save_path+name, res)
