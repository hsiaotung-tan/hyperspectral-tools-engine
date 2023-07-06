import torch
import threading

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()

class AddGaussanNoiseStd(object):
    """add gaussian noise to the given torch tensor (B,H,W)"""
    def __init__(self, sigma):
        self.sigma_ratio = sigma / 255.
    def __call__(self, img):
        noise = torch.randn_like(img) * self.sigma_ratio
        img += noise
        return img
    
class AddGaussanBlindNoiseStd(object):
    """add gaussian noise to the given torch tensor (B,H,W)"""
    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, sigmas):
        self.sigmas_ratio = torch.tensor(sigmas) / 255.
        self.pos = LockedIterator(self.__pos(len(sigmas)))

    def __call__(self, img):
        noise = torch.randn_like(img) * self.sigmas_ratio[next(self.pos)]
        img += noise
        return img
    
if __name__ == '__main__':
    t = AddGaussanBlindNoiseStd(sigmas=[15, 30, 50, 70])
    img = torch.randn((1, 3, 31, 31))
    print(t(img).shape)