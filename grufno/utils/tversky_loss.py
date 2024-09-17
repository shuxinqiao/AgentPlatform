import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    """
    Compute the Tversky loss defined in:

        Sadegh et al. (2017) Tversky loss function for image segmentation
        using 3D fully convolutional deep networks. (https://arxiv.org/abs/1706.05721)

    Args:
        alpha (float): alpha parameter in original equation.
        beta (float): beta parameter in original equation.
        eps (float): preventing inf error.
        threshold (float): binary mask threshold, over is 1, under is 0.
        dims (tuple): dimensions order for averaging.

    Returns:
        torch.Tensor: averaged binary tversky loss.
    """
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6, threshold=0.01, dims=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.threshold = threshold
        self.dims = dims
        
    def custom_flatten(self, x, start_dim):
        # Get the shape of the input tensor
        shape = x.shape

        # Calculate the number of elements to be flattened
        num_elements = 1
        for dim in shape[start_dim:]:
            num_elements *= dim

        # Reshape the tensor using view
        return x.view(*shape[:start_dim], num_elements)


    def forward(self, y_pred, y_true):
        """
        Compute Binary Tversky loss between predicted and ground truth.

        Args:
            y_pred (torch.Tensor): Predicted.
            y_true (torch.Tensor): Ground truth.

        Returns:
            torch.Tensor: Tversky loss.
        """

        # Flatten predictions and ground truth
        y_pred = self.custom_flatten(y_pred, start_dim=self.dims)
        y_true = self.custom_flatten(y_true, start_dim=self.dims)

        y_pred = torch.abs(y_pred) >= self.threshold
        y_true = torch.abs(y_true) >= self.threshold
        
        # Compute true positives, false positives, and false negatives
        tp = torch.sum(y_true * y_pred, dim=2)
        fp = torch.sum(y_pred, dim=2) - tp
        fn = torch.sum(y_true, dim=2) - tp

        # Compute Tversky coefficient
        tversky_coeff = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)

        # Compute Tversky loss
        tversky_loss = 1 - tversky_coeff
        #print(tversky_loss)
        # Average over specified dimensions
        tversky_loss = torch.mean(tversky_loss)

        return tversky_loss