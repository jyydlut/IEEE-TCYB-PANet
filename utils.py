import torch
import torch.nn.functional as F
from torch.autograd import Variable
Mat16 = torch.zeros(16, 1)
Mat16T = torch.zeros(1, 16)
for i in range(16):
  Mat16[i, 0] = i + 1
  Mat16T[0, i] = i + 1
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=40):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
def loss_position(step, out, gt):
  b = out.size()[0]
  Mat16b, Mat16Tb = Variable(Mat16.cuda()), Variable(Mat16T.cuda())
  Mat16b, Mat64Tb = Mat16b.repeat(b, 1, 1), Mat16Tb.repeat(b, 1, 1)
  loss_mse= torch.nn.MSELoss()
  step = int(pow(2, step))
  shape = step*16
  out = F.interpolate(out, size=(shape, shape), mode='bilinear', align_corners=False)
  gt = F.interpolate(gt, size=(shape, shape), mode='bilinear', align_corners=False)
  out, gt = out.view(b, shape, shape), gt.view(b, shape, shape)
  x, xo = torch.zeros((b, step, step)), torch.zeros((b, step, step))
  y, yo = torch.zeros((b, step, step)), torch.zeros((b, step, step))
  x, xo, y, yo = x.cuda(), xo.cuda(), y.cuda(), yo.cuda()
  for i in range(step):
    for j in range(step):
      term = gt[:, i*16:i*16+16, j*16:j*16+16].view(b, 16, 16)
      termo = out[:, i*16:i*16+16, j*16:j*16+16].view(b, 16, 16)
      t, to = torch.sum(term, dim=(1, 2)), torch.sum(termo, dim=(1, 2))
      t, to = 1+t, 1+to
      x[:, i, j], y[:, i, j] = torch.sum(torch.bmm(Mat16Tb, term), dim=(1, 2)) / t, torch.sum(torch.bmm(term, Mat16b), dim=(1, 2)) / t
      xo[:, i, j], yo[:, i, j] = torch.sum(torch.bmm(Mat16Tb, termo), dim=(1, 2)) / to, torch.sum(torch.bmm(termo, Mat16b), dim=(1, 2)) / to
  x, xo, y, yo = x/16.0, xo/16.0, y/16.0, yo/16.0
  loss = loss_mse(xo, x)+loss_mse(yo, y)
  return loss
def loss_boundary(out, gt):
    def boundary(input):
        avgpool = F.avg_pool2d(input, kernel_size=3, stride=1, padding=1)
        pavgpool = 1 - F.avg_pool2d(1 - input, kernel_size=3, stride=1, padding=1)
        out = avgpool - pavgpool
        return out
    loss_mse = torch.nn.MSELoss()
    outb = boundary(out)
    gtb = boundary(gt)
    return loss_mse(outb, gtb)

