import torch
import torch.nn.functional as F
from datetime import datetime
#from torch.autograd import Variable
Mat16, Mat16T = torch.zeros((16, 1)), torch.zeros((1, 16))
mse = torch.nn.MSELoss(size_average=True)
for i in range(16):
    Mat16[i, 0] = i+1
    Mat16T[0, i] = i+1

def cal_pos(input, num):
    ba, c, w, h = input.size()
    M, MT = Mat16.repeat(ba, 1, 1).cuda(), Mat16T.repeat(ba, 1, 1).cuda()
    x, y = torch.zeros((ba, num, num)).cuda(), torch.zeros((ba, num, num)).cuda()
    input = F.interpolate(input, size=(16*num, 16*num), mode='bilinear')
    for i in range(num):
        for j in range(num):
            term = input[:, :, i*16:i*16+16, j*16:j*16+16].view(ba, 16, 16)
            t = torch.sum(term, dim=(1, 2))+1
            x[:, i, j], y[:, i, j] = torch.sum(torch.bmm(MT, term), dim=(1, 2)) / t, torch.sum(torch.bmm(term, M), dim=(1, 2)) / t
    return x, y
def pos_gt(focals, rgbs, num):
    ba, c, w, h = rgbs.size()
    px, py = cal_pos(rgbs, num)
    fpx, fpy = cal_pos(focals, num)
    px, py = px.repeat(12, 1, 1), py.repeat(12, 1, 1)
    t = (fpx-px)**2+(fpy-py)**2
    t = t.view(ba*12, 1, num, num)
    t = torch.cat(torch.chunk(t, 12, dim=0), dim=1)
    out = torch.argmin(t, dim=1)
    return out
def sal_gt(focals, rgbs, num):
    ba, c, w, h = rgbs.size()
    focals = F.interpolate(focals, size=(num*16, num*16), mode='bilinear')
    rgbs = F.interpolate(rgbs, size=(num * 16, num * 16), mode='bilinear')
    rgbs = rgbs.repeat(12, 1, 1, 1)
    out = torch.zeros((ba*12, num, num))
    for i in range(num):
        for j in range(num):
            out[:, i, j] = torch.sum(focals[:, :, 16*i:16*(i+1), 16*j:16*(j+1)]-rgbs[:, :, 16*i:16*(i+1), 16*j:16*(j+1)], dim=(1,2, 3))
    out = out.view(ba*12, 1, num, num)
    out = torch.cat(torch.chunk(out, 12, dim=0), dim=1)
    out = torch.argmin(out, dim=1)
    return out
def boundary_gt(focals, rgbs, num):
    def boundary(input):
        avgpool = F.avg_pool2d(input, kernel_size=3, stride=1, padding=1)
        pavgpool = 1 - F.avg_pool2d(1 - input, kernel_size=3, stride=1, padding=1)
        out = avgpool - pavgpool
        return out
    ba, c, w, h = rgbs.size()
    focals = F.interpolate(focals, size=(num*16, num*16), mode='bilinear')
    rgbs = F.interpolate(rgbs, size=(num * 16, num * 16), mode='bilinear')
    rgbs = rgbs.repeat(12, 1, 1, 1)
    focals, rgbs = boundary(focals), boundary(rgbs)
    out = torch.zeros((ba*12, num, num))
    for i in range(num):
        for j in range(num):
            out[:, i, j] = torch.sum(focals[:, :, 16*i:16*(i+1), 16*j:16*(j+1)]-rgbs[:, :, 16*i:16*(i+1), 16*j:16*(j+1)], dim=(1,2, 3))
    out = out.view(ba*12, 1, num, num)
    out = torch.cat(torch.chunk(out, 12, dim=0), dim=1)
    out = torch.argmin(out, dim=1)
    return out
def fusedidx(xf,x2f, xr, x2r, mslm, srm, num):
    ba = xr.size()[0]
    t1 = mslm(xf, xr, num)
    t2, l1, l2 = srm(x2f, x2r, num)
    t2p = t2.view(12*ba, 1, num, num)
    t2p = torch.cat(torch.chunk(t2p, 12, dim=0), dim=1)
    avg_t2 = torch.mean(t2p, dim=1)
    max_t2 = torch.max(t2p, dim=1)[0]
    t2_avg = avg_t2.repeat(12, 1, 1)
    max_v = max_t2.repeat(12, 1, 1)
    t2[t2>=t2_avg]=1
    t2[t2<t2_avg]=max_v[t2<t2_avg]-t2[t2<t2_avg]
    t = t1*t2
    t = t.view(ba*12, 1, num, num)
    t = torch.cat(torch.chunk(t, 12, dim=0), dim=1)
    out = torch.argmax(t, dim=1)
    return out
def fused(feature, num, arg):
    ba, c, w, h = feature.size()
    b = int(ba/12)
    s = int(w/num)
    feature = feature.view(ba, 1, c, w, h)
    feature = torch.cat(torch.chunk(feature, 12, dim=0), dim=1)
    out = torch.zeros((b, c, w, h))
    for i in range(num):
        for j in range(num):
            out[:, :, s * i:s * (i + 1), s * j:s * (j + 1)] = feature[torch.range(0, b-1).long(), arg[:, i, j], :, s*i:s*(i+1), s*j:s*(j+1)]
    return out