import torch

# normalization, pointwise gaussian
class TimeGaussianNormalizer(object):
    def __init__(self, x):
        super(TimeGaussianNormalizer, self).__init__()

        # Init (B,T,C,X,Y,Z) keeping Time and Channel
        self.mean = torch.mean(x,(0,3,4,5))[None,:,:,None,None,None]
        self.std = torch.std(x,(0,3,4,5))[None,:,:,None,None,None]

    def encode(self, x):
        x = (x - self.mean)/self.std
        return x

    def decode(self, x):
        x = (x * self.std) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        
    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()