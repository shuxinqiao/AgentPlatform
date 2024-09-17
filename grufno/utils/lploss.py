import torch
    
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, LGR=False, mode='sat'):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.LGR = LGR

        self.mode = mode

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def abs_sat(self, x, y, threshold=0.01):
        num_examples = x.size()[0]

        # Apply threshold
        threshold_mask = torch.logical_or(x > threshold, y > threshold)

        x_thresh = x[threshold_mask]
        y_thresh = y[threshold_mask]

        # Calculate the number of elements that exceeded the threshold
        num_masked_elements = torch.sum(threshold_mask)

        all_norms = torch.norm(x_thresh.reshape(num_examples,-1) - y_thresh.reshape(num_examples,-1), 1, 1) / num_masked_elements


        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y, mode):
        num_examples = x.size()[0]
        num_timestep = x.size()[1]

        if mode == "sat":
            diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
            y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        elif mode == "pres":
            #diff_norms = torch.norm(x.reshape(num_examples,num_timestep,-1) - y.reshape(num_examples,num_timestep,-1), self.p, 2)
            diff_norms = torch.abs(x.reshape(num_examples,num_timestep,-1) - y.reshape(num_examples,num_timestep,-1))
            y_norms = torch.amax(y, dim=(2,3,4,5), keepdim=True)
            #y_norms = torch.amax(y.reshape(num_examples,num_timestep,-1), self.p, 2)
            #return torch.mean(diff_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms
    
    
    def global_loss(self, x, y, glob):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/glob)
            else:
                return torch.sum(diff_norms/glob)

        return diff_norms/glob

    def __call__(self, x, y, glob=None):
        if self.LGR is False:
            return self.rel(x, y, self.mode)
        else:
            return self.global_loss(x, y, glob)
    
    