import os
import cv2
class data_aug:
    def __init__(self, image_root, gt_root, out_root):
        self.augument = ['', '_c', '_rt', '_lb', '_lt', '_rb', '_flr', '_ftb', '_ro90', '_ro180', '_ro270']
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        sorted(self.images)
        sorted(self.gts)
        self.out_root = out_root
    def crop(self, x, mode):
        if mode == 0:
            img = cv2.resize(x, None, fx=1 / 0.9, fy=1 / 0.9, interpolation=cv2.INTER_CUBIC)
            sp = x.shape
            x, y = int((sp[0] - 256) / 2), int((sp[1] - 256) / 2)
            cropImg = img[x:x + 256, y:y + 256]
            return cropImg
        if mode == 1:
            img = cv2.resize(x, None, fx=1, fy=1 / 0.9, interpolation=cv2.INTER_CUBIC)
            sp = x.shape
            x, y = 0, 0
            cropImg = img[x:x + 256, y:y + 256]
            return cropImg
        if mode == 2:
            img = cv2.resize(x, None, fx=1 / 0.9, fy=1, interpolation=cv2.INTER_CUBIC)
            sp = x.shape
            x, y = 0, 0
            cropImg = img[x:x + 256, y:y + 256]
            return cropImg
        if mode == 3:
            img = cv2.resize(x, None, fx=1, fy=1 / 0.9, interpolation=cv2.INTER_CUBIC)
            sp = x.shape
            x, y = int(sp[0] - 256), 0
            cropImg = img[x:x + 256, y:y + 256]
            return cropImg
        if mode == 4:
            img = cv2.resize(x, None, fx=1, fy=1 / 0.9, interpolation=cv2.INTER_CUBIC)
            sp = x.shape
            x, y = 0, int(sp[0] - 256)
            cropImg = img[x:x + 256, y:y + 256]
            return cropImg
        return x
    def rotate(self, x, degree):
        if degree == 90:
            return cv2.rotate(x, cv2.cv2.ROTATE_90_CLOCKWISE)
        if degree == 180:
            return cv2.rotate(x, cv2.cv2.ROTATE_180)
        if degree == 270:
            return cv2.rotate(x, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        return x
    def flip(self, x, mode):
        if mode == 1:
            return cv2.flip(x, 1)
        if mode == 2:
            return cv2.flip(x, 0)
        return x
    def aug(self, x, mode):
        if mode == 0: return x
        if mode >= 1 and mode <= 5: return self.crop(x, mode - 1)
        if mode >= 6 and mode <= 7: return self.flip(x, mode - 5)
        if mode >= 8 and mode <= 10: return self.rotate(x, (mode - 7) * 90)
        return x
    def forward(self):
        for i in range(len(self.images)):
            img = cv2.imread(self.images[i])
            gt = cv2.imread(self.gts[i])
            name = self.images[i].split('/')[-1][:-4]
            for j in range(len(self.augument)):
                imgname = out_root + '/train_images/' + name + self.augument[j] + '.jpg'
                img = self.aug(img, j)
                gtname = out_root + '/train_gts/' + name + self.augument[j] + '.png'
                gt = self.aug(gt, j)
                cv2.imwrite(imgname, img)
                cv2.imwrite(gtname, gt)
if __name__=='__main__':
    image_root = 'root of images before augument'
    gt_root = 'root of gts before augument'
    out_root = 'saveing root'
    data_aug = data_aug(image_root, gt_root, out_root)
    data_aug.forward()
