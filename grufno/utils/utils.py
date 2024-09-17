########################################
# Data helper functions
########################################
import h5py
import numpy as np

# Helper function - Summarize hdf5 file structure
def print_hdf5_file_structure(file):
    def print_attrs(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print("    Shape:", obj.shape)

        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    with h5py.File(file, "r") as f:
        f.visititems(print_attrs)

def load_data(data_path, array_name_list, chunck=None):
    hf_r = h5py.File(data_path, 'r')
    result = []
    for name in array_name_list:
        if chunck == None:
            result.append(np.array(hf_r.get(name)))
        else:
            result.append(np.array(hf_r.get(name)[chunck[0]:chunck[1]]))
    hf_r.close()
    return result


########################################
# Training log functions
########################################
import os
import sys
import datetime
import torch

def iter_log(work_dirs, filename, epoch, batch, loss):
    # Open the log file in append mode
    with open(work_dirs+filename, "a") as log_file:    
        
        # Training steps (epoch, batch, loss)
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
        #loss_result = str(loss).replace("{","").replace("}", "")
        formatted_losses = ', '.join([f'{key}: {value}' for key, value in loss.items()])

        log_file.write(f"Time: {timestamp}, Epoch: {epoch}, Batch: {batch}, {formatted_losses}\n")
        print(f"Epoch: {epoch}, Batch: {batch}, {formatted_losses}\n")

        # Add a separator for better readability
        #log_file.write("\n" + "-" * 20 + "\n")

def save_log(work_dirs, filename, name):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Open the log file in append mode
    with open(work_dirs+filename, "a") as log_file:    
        # Save information (if provided)
        log_file.write(f"\nModel Saved: ")
        log_file.write(f"Time: {current_time}\t")
        log_file.write(f"Name: {name}\n\n")

def init_log(work_dirs, filename, model, detail=None):
    if not os.path.exists(work_dirs):
        # If it doesn't exist, create it
        os.makedirs(work_dirs)

    # Open the log file in append mode
    with open(work_dirs+filename, "a") as log_file:
        # Python, CUDA, and Torch versions
        log_file.write(f"Python Version: {sys.version}\n")
        log_file.write(f"CUDA Availability: {torch.cuda.is_available()}\n")
        log_file.write(f"PyTorch Version: {torch.__version__}\n")

        # Details (in dict)
        if detail:
            log_file.write(', '.join([f'{key}: {value}' for key, value in detail.items()]))
            log_file.write("\n")


        # Add a separator for better readability
        log_file.write("\n" + "-" * 20 + "\n")

        # Model structure
        log_file.write("Model Structure:\n")
        log_file.write(f"{model}")
        # Add a separator for better readability
        log_file.write("\n" + "-" * 20 + "\n")

        



########################################
# Relative Error functions
########################################
from torch import nn

def relative_error(model_output, ground_truth, mode):
    """
    Relative error from 3DRRUET paper
    Accepting 3 modes: saturation, pressure, displacement
    Epsilon = 0.01

    args:
        model_output: 6D tensor (B,T,C,D,H,W)
        ground_truth: 6D tensor (B,T,C,D,H,W)
        mode: ['sat', 'pres', 'disp']
    """
    if mode == 'sat':
        eps = 0.01
        return torch.mean(torch.abs(model_output - ground_truth) / torch.abs((ground_truth + eps)))
    elif mode == "sat_nest":
        mask = (torch.abs(model_output) > 0.01) | (torch.abs(ground_truth) > 0.01)
        return torch.sum(torch.abs(ground_truth - model_output)[mask == 1]) / torch.sum(mask)
    elif mode == 'pres' or mode == "pressure":
        # Assuming (B,T,C,D,H,W) and C is 1
        max_vals = torch.amax(ground_truth, dim=(2,3,4,5), keepdim=True)
        min_vals = torch.amin(ground_truth, dim=(2,3,4,5), keepdim=True)
        diff = max_vals# - min_vals

        abs_error = torch.abs(model_output - ground_truth)

        return torch.mean(abs_error / diff)
    elif mode == 'disp':
        return torch.mean(torch.abs(model_output - ground_truth) / (ground_truth))


def intersection_over_union(y_pred, y_true, eps=1e-6, threshold=0.01, dims=2):
    """
    Compute Intersection over Union (IoU) between predicted and ground truth.

    Args:
        y_pred (torch.Tensor): Prediction.
        y_true (torch.Tensor): Ground truth.
        threshold (float): Threshold value for binarization.

    Returns:
        torch.Tensor: Intersection over Union (IoU) metric.
    """
    # Flatten predictions and ground truth
    y_pred = torch.flatten(y_pred, start_dim=dims)
    y_true = torch.flatten(y_true, start_dim=dims)
    
    # Threshold masks
    y_pred = y_pred >= threshold
    y_true = y_true >= threshold

    # Compute intersection and union
    intersection = torch.sum(y_true * y_pred, dim=2)
    union = torch.sum((y_true + y_pred) > 0, dim=2)

    # Compute IoU
    iou = (intersection + eps) / (union + eps)
    
    iou = torch.mean(iou)

    return iou


########################################
# Dataset Class
########################################
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path, x_name_list, y_name, transform=None):
        self.hf_r = h5py.File(hdf5_file_path, 'r')
        
        self.data_x_list = []
        for x_channel in x_name_list: 
            self.data_x_list.append(self.hf_r[x_channel])

        self.data_x = np.concatenate(self.data_x_list, axis=2)

        self.data_y = self.hf_r[y_name]
        self.transform = transform

    def __len__(self):
        return self.data_y.shape[0]  # Assuming N is the first dimension

    def __getitem__(self, idx):
        sample_x = self.data_x[idx, ...]  # Extract a single item for x
        sample_y = self.data_y[idx, ...]  # Extract the corresponding item for y
        
        sample_x = torch.tensor(sample_x, dtype=torch.float32)
        sample_y = torch.tensor(sample_y, dtype=torch.float32)
        
        if self.transform is not None:
            sample_x, sample_y = self.transform(sample_x, sample_y)

        return sample_x, sample_y
    


########################################
# 3D plot helper
########################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

def plot_3d_image(data, batch=0, time=0, channel=0, show_legend=True, value_range=None, figsize=(8,6), dpi=100):
    """
    Plot a 3D image from input data with specified batch, time, and channel.
    
    Parameters:
        data (numpy.ndarray): Input data with shape (Batch, Time, Channel, D, H, W).
        batch (int): Index of the batch dimension.
        time (int): Index of the time dimension.
        channel (int): Index of the channel dimension.
        show_legend (bool): Whether to show the color gradient legend. Default is True.
        value_range (tuple): Tuple of minimum and maximum values to display.
                             If None, the entire range of values will be displayed.
        figsize (tuple): Width and height of the figure in inches. Default is (10, 8).
        dpi (int): Dots per inch (resolution) of the figure. Default is 100.
    """
    # Extract the specified data slice
    slice_data = data[batch, time, channel]
    
    # Determine value range for transparency
    if value_range is None:
        value_min = np.min(slice_data)
        value_max = np.max(slice_data)
    else:
        value_min, value_max = value_range
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid representing voxel indices
    z, x, y = np.indices(tuple(x+1 for x in slice_data.shape))
    z = z[::-1]
    
    # Normalize data to map to colors
    norm = Normalize(vmin=np.min(slice_data), vmax=np.max(slice_data))
    
    # Mask values outside the specified range for transparency
    masked_data = np.where((slice_data >= value_min) & (slice_data <= value_max), slice_data, 0)

    masked_color = np.where((slice_data >= value_min) & (slice_data <= value_max), norm(slice_data), np.nan)
            

    # Plot the voxels with gradient color and transparency
    ax.voxels(x, y, z, masked_data, facecolors=plt.cm.autumn(norm(slice_data)), edgecolors=None, shade=False)    
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Image (Batch: {batch}, Time: {time}, Channel: {channel})')
    
    # Add color legend if specified
    if show_legend:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.autumn, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Value')
    
    
    plt.show()



########################################
# Image Uniform Smoothing
########################################

def uniform_filter_3d(image, kernel_size=3):
    # Create a 3D uniform kernel
    kernel = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.float32) / (kernel_size**3)
    
    # Pad the image to handle edges
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((0, 0), (0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    
    # Initialize the smoothed image
    smoothed_image = np.zeros_like(image)
    
    # Perform convolution only on Z, X, Y dimensions
    for t in range(image.shape[1]):
        for z in range(image.shape[3]):
            for x in range(image.shape[4]):
                for y in range(image.shape[5]):
                    smoothed_image[0, t, 0, z, x, y] = np.sum(padded_image[0, t, 0, z:z+kernel_size, x:x+kernel_size, y:y+kernel_size] * kernel)
    
    return smoothed_image


def uniform_filter_2d(image, kernel_size=3):
    # Create a uniform kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    # Pad the image to handle edges
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    # Initialize the smoothed image
    smoothed_image = np.zeros_like(image)
    
    # Perform convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            smoothed_image[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return smoothed_image