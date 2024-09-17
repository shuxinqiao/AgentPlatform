import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    """
    3D convolution block.
    Consists 3 layers:
        1. Conv3d
        2. BatchNorm
        3. ReLU

    Args:
        input_channels,
        output_channels,
        kernel_size: int or tuple(a,b,c),
        stride: int or tuple(a,b,c)
    """

    def __init__(self, input_channels, output_channels, kernel_size=(3,3,3), stride=(1,1,1)):
        super().__init__()

        # Instance layers
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding=1)
        self.bn = nn.BatchNorm3d(output_channels)#, eps=0.001, momentum=0.9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    

class res_block(nn.Module):
    """
    3D residual convolution block.
    Consists 5 layers:
        1. Conv3d_1
        2. BatchNorm_1
        3. ReLU
        4. Conv3d_2
        5. BatchNorm_2

    Args:
        input_channels,
        output_channels,
        kernel_size: int or tuple(a,b,c),
        stride: int or tuple(a,b,c)
    """

    def __init__(self, input_channels, output_channels, kernel_size=(3,3,3), stride=(1,1,1)):
        super().__init__()

        # Instance layers
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm3d(output_channels)#, eps=0.001, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding=1)
        self.bn2 = nn.BatchNorm3d(output_channels)#, eps=0.001, momentum=0.9)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return torch.add(identity, x)
    

class repeat_conv(nn.Module):
    """
    Repeats the input n times.
   
    # Example
    ```
    pytorch:
        model = Sequential(
            nn.Linear(in_features=16, out_features=32)
        )
        # now: model.output_shape == (None, 32)
        # note: `None` is the batch dimension
        model = Sequential(
            nn.Linear(in_features=32, out_features=32)
            repeat_conv(3)
        )
        # now: model.output_shape == (None, 3, 32)
    ```

    Args
        depth: integer, repetition factor.
        Input shape: 5D tensor of shape `(batch, C, D, H, W)`.
        Output shape: 5D tensor of shape `(batch, T, C, D, H, W)`.
    """

    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def forward(self, x):
        # Repeat the input tensor along the new Time Step dimension
        x = x.unsqueeze(1).repeat(1, self.depth, 1, 1, 1, 1)
        
        return x


class TimeDistributed(nn.Module):
    """
    Applies `module` over `tdim` identically for each step, 
    use `low_mem` to compute one at a time.
    """
    def __init__(self, module, low_mem=False, tdim=1):
        super().__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim
        
    def forward(self, tensors):
        "input x with shape:(bs,seq_len,channels,depth,width,height)"
        
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(tensors)
        else:
            #only support tdim=1
            inp_shape = tensors.shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(tensors.view(bs*seq_len, *tensors.shape[2:]))
        return self.format_output(out, bs, seq_len)
    
    def low_mem_forward(self, tensors):                                           
        "input x with shape:(bs,seq_len,channels,depth,width,height)"
        
        seq_len = tensors.shape[self.tdim]
        tensor_split = torch.unbind(tensors, dim=self.tdim) # tuple

        out = []
        for i in range(seq_len):
            out.append(self.module(tensor_split[i]))
        
        return torch.stack(out, dim=self.tdim)
    
    def format_output(self, out, bs, seq_len):
        "unstack from batchsize outputs"
        if isinstance(out, tuple):
            return tuple(out_i.view(bs, seq_len, *out_i.shape[1:]) for out_i in out)
        return out.view(bs, seq_len,*out.shape[1:])
    
    
    def range_of(self, x):
        "Create a range from 0 to `len(x)`."
        return list(range(len(x)))

    def __repr__(self):
        return f'TimeDistributed({self.module})'


class time_conv_block(nn.Module):
    """
    3D residual convolution block with Time wrapped.
    Consists 5 layers:
        1. Conv3d_1
        2. BatchNorm_1
        3. ReLU
        4. Conv3d_2
        5. BatchNorm_2

    Args:
        input_channels,
        output_channels,
        kernel_size: int or tuple(a,b,c),
        stride: int or tuple(a,b,c)
    """
    
    def __init__(self, input_channels, output_channels, kernel_size=(3,3,3), stride=(1,1,1), tdim=1):
        super().__init__()
        self.tdim = tdim

        # Instance layers
        self.conv = TimeDistributed(nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding=1), low_mem=True, tdim=self.tdim)
        self.bn = TimeDistributed(nn.BatchNorm3d(output_channels), low_mem=True, tdim=self.tdim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    


class time_res_block(nn.Module):
    """
    3D residual convolution block.
    Consists 5 layers with timedistributed:
        1. Conv3d_1
        2. BatchNorm_1
        3. ReLU
        4. Conv3d_2
        5. BatchNorm_2

    Args:
        input_channels,
        output_channels,
        kernel_size: int or tuple(a,b,c),
        stride: int or tuple(a,b,c)
    """

    def __init__(self, input_channels, output_channels, kernel_size=(3,3,3), stride=(1,1,1), tdim=1):
        super().__init__()
        self.tdim = tdim

        # Instance layers
        self.conv1 = TimeDistributed(nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding=1), low_mem=True, tdim=self.tdim)
        self.bn1 = TimeDistributed(nn.BatchNorm3d(output_channels), low_mem=True, tdim=self.tdim)
        self.relu = TimeDistributed(nn.ReLU(), low_mem=True, tdim=self.tdim)
        self.conv2 = TimeDistributed(nn.Conv3d(input_channels, output_channels, kernel_size, stride, padding=1), low_mem=True, tdim=self.tdim)
        self.bn2 = TimeDistributed(nn.BatchNorm3d(output_channels), low_mem=True, tdim=self.tdim)

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return torch.add(identity, x)
    
    
class time_deconv_block(nn.Module):
    """
    3D deconvolution block.
    Consists 5 layers with timedistributed:
        1. Upsampling3d
        2. Conv3d
        3. BatchNorm
        4. ReLU

    Args:
        input_channels,
        output_channels,
        kernel_size: int or tuple(a,b,c),
        stride: int or tuple(a,b,c)
    """

    def __init__(self, input_channels, output_channels, kernel_size=(3,3,3), stride=(2,2,2), tdim=1):
        super().__init__()
        self.tdim = tdim

        # Instance layers
        self.upsample = TimeDistributed(nn.Upsample(scale_factor=stride), low_mem=True, tdim=self.tdim)
        self.conv = TimeDistributed(nn.Conv3d(input_channels, output_channels, kernel_size, stride=1, padding=1, padding_mode='reflect'), low_mem=True, tdim=self.tdim)
        self.bn = TimeDistributed(nn.BatchNorm3d(output_channels), low_mem=True, tdim=self.tdim)
        self.relu = TimeDistributed(nn.ReLU(), low_mem=True, tdim=self.tdim)

    def forward(self, x):
        
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    

class time_LFS_output_block(nn.Module):
    """
    Final output 3D deconvolution block with low resolution for training.
    Consists 2 layers with timedistributed:
        1. Upsampling3d (It works in both up and down direction)
        2. Conv3d

    Args:
        input_channels,
        output_channels,
        kernel_size: int or tuple(a,b,c),
        stride: int or tuple(a,b,c)
    """
    
    def __init__(self, input_channels, output_channels, output_size, kernel_size=(3,3,3), stride=(2,2,2), tdim=1):
        super().__init__()
        self.tdim = tdim

        # Instance layers
        self.upsample = TimeDistributed(nn.Upsample(size=output_size), low_mem=True, tdim=self.tdim)
        self.conv = TimeDistributed(nn.Conv3d(input_channels, output_channels, kernel_size, stride=1, padding=1, padding_mode='zeros'), low_mem=True, tdim=self.tdim)

    def forward(self, x):
        
        x = self.upsample(x)
        x = self.conv(x)

        return x
    


class time_HFS_output_block(nn.Module):
    """
    Final output 3D deconvolution block with high resolution for training.
    Consists 2 layers with timedistributed:
        1. Upsampling3d (It works in both up and down direction)
        2. Conv3d

    Args:
        input_channels,
        output_channels,
        kernel_size: int or tuple(a,b,c),
        stride: int or tuple(a,b,c)
    """
    
    def __init__(self, input_channels, output_channels, output_size, kernel_size=(3,3,3), stride=(2,2,2), tdim=1):
        super().__init__()
        self.tdim = tdim

        # Instance layers
        #self.upsample = TimeDistributed(nn.Upsample(size=output_size), low_mem=True, tdim=self.tdim)
        self.conv = TimeDistributed(nn.Conv3d(input_channels, output_channels, kernel_size, stride=1, padding=1, padding_mode='zeros'), low_mem=True, tdim=self.tdim)

    def forward(self, x):
        
        #x = self.upsample(x)
        x = self.conv(x)

        return x