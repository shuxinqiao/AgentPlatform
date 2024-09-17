import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from grufno.utils.layers import TimeDistributed


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    

class Cell(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, skip, activation=True):
        super(Cell, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.skip = skip
        self.acticvation = activation

        self.conv_x = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp_x = MLP(self.width, self.width, self.width)
        self.w_x = nn.Conv3d(self.width, self.width, 1)
        
        self.conv_h = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp_h = MLP(self.width, self.width, self.width)
        self.w_h = nn.Conv3d(self.width, self.width, 1)

        # Reset gate
        self.reset_gate = nn.Conv3d(self.width*2, self.width, kernel_size=3, padding=1)
        # Update gate
        self.update_gate = nn.Conv3d(self.width*2, self.width, kernel_size=3, padding=1)
        # Output gate
        self.out_gate = nn.Conv3d(self.width*2, self.width, kernel_size=3, padding=1)

        # LayerNorms
        #C, D, H, W = self.width, 24+4*2, 60+4*2, 60+4*2
        #self.LN_1 = nn.LayerNorm([C, D, H, W])
        #self.LN_2 = nn.LayerNorm([C, D, H, W])
        #self.LN_3 = nn.LayerNorm([C, D, H, W])


    def forward(self, x):
        # Input Shape: (B, T, C, D, H, W)
        time_size = x.shape[1]

        outputs_list = []
        
        for t in range(time_size):
            cur_inp = x[:,t,...].to(x.device) # (B,C,D,H,W)
            
            if t == 0:  # zero initial
                prev_state = torch.zeros(*cur_inp.shape).to(x.device)
             
             
            # Compute gates
            combined = torch.cat([cur_inp, prev_state], dim=1)  # Concatenate along channel dimension
            #reset_gate = self.LN_1(self.reset_gate(combined))
            reset_gate = self.reset_gate(combined)
            reset_gate = torch.sigmoid(reset_gate)

            #update_gate = self.LN_2(self.update_gate(combined))
            update_gate = self.update_gate(combined)
            update_gate = torch.sigmoid(update_gate)

            combined_reset = torch.cat([cur_inp, reset_gate * prev_state], dim=1)

            #out_gate = self.LN_3(self.out_gate(combined_reset))
            out_gate = self.out_gate(combined_reset)
            out_gate = torch.tanh(out_gate)

            prev_state = update_gate * prev_state + (1 - update_gate) * out_gate
            #prev_state = self.LN_1(prev_state)

            # current input
            c_x1 = self.conv_x(cur_inp)
            c_x1 = self.mlp_x(c_x1)
            c_x2 = self.w_x(cur_inp)

            if self.skip:
                c = c_x1 + c_x2 + cur_inp
            else:
                c = c_x1 + c_x2
                
            c = F.gelu(c)

            h_x1 = self.conv_h(prev_state)
            h_x1 = self.mlp_h(h_x1)
            h_x2 = self.w_h(prev_state)

            if self.skip:
                h = h_x1 + h_x2 + cur_inp
            else:
                h = h_x1 + h_x2
            
            h = F.gelu(h)

            prev_state = h + c

            if self.acticvation:
                #prev_state = F.gelu(h + c)
                outputs_list.append(F.gelu(prev_state))
            else:
            #    prev_state = h + c
                outputs_list.append(prev_state)

            #outputs_list.append(prev_state)


        x = torch.stack(outputs_list, dim=1) # Stack time outputs in dim=1, now (B,T,C,D,H,W)
        
        return x

    

class LSKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0z = TimeDistributed(nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), stride=(1,1,1), padding=((3-1)//2, 0, 0), groups=dim), True, 1)
        self.conv0x = TimeDistributed(nn.Conv3d(dim, dim, kernel_size=(1, 3, 1), stride=(1,1,1), padding=(0, (3-1)//2, 0), groups=dim), True, 1)
        self.conv0y = TimeDistributed(nn.Conv3d(dim, dim, kernel_size=(1, 1, 3), stride=(1,1,1), padding=(0, 0, (3-1)//2), groups=dim), True, 1)

        self.conv_spatial_z = TimeDistributed(nn.Conv3d(dim, dim, kernel_size=(5, 1, 1), stride=(1,1,1), padding=(4, 0, 0), groups=dim, dilation=2), True, 1)
        self.conv_spatial_x = TimeDistributed(nn.Conv3d(dim, dim, kernel_size=(1, 5, 1), stride=(1,1,1), padding=(0, 4, 0), groups=dim, dilation=2), True, 1)
        self.conv_spatial_y = TimeDistributed(nn.Conv3d(dim, dim, kernel_size=(1, 1, 5), stride=(1,1,1), padding=(0, 0, 4), groups=dim, dilation=2), True, 1)

        self.point_conv = TimeDistributed(nn.Conv3d(dim, dim, 1), True, 1)

    def forward(self, x):
        res = x.clone()

        x = self.conv0z(x)
        x = self.conv0x(x)
        x = self.conv0y(x)

        x = self.conv_spatial_z(x)
        x = self.conv_spatial_z(x)
        x = self.conv_spatial_z(x)

        x = self.point_conv(x)

        return res * x

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.DW_conv = TimeDistributed(nn.Conv3d(dim, dim, 5, padding=2, groups=dim), True, 1)
        self.DW_D_conv = TimeDistributed(nn.Conv3d(dim, dim, 5, padding=8, dilation=4, groups=dim), True, 1)
        self.point_conv = TimeDistributed(nn.Conv3d(dim, dim, 1), True, 1)

    def forward(self, x):
        res = x

        x = self.DW_conv(x)
        x = self.DW_D_conv(x)
        x = self.point_conv(x)

        return res * x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, input_channel=1, num_cell=4, skip=False, periodic=False):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier-RNN cell.
        1. Pad z,x,y dimensions with zeros
        1. Lift the input channel dimension to the desire size by self.width.
        2. 4 layers of Fourier-RNN cell
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input shape: (batchsize, Time, Channel(default=1), Depth, Height, Width)
        output shape: (batchsize, Time, Channel(default=1), Depth, Height, Width)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 4 # pad the domain if input is non-periodic
        self.skip = skip
        self.periodic = periodic

        self.p = nn.Linear(input_channel, self.width)   # input channel is 1

        self.num_cell = num_cell

        self.cells = nn.ModuleList()
        self.LKAs = nn.ModuleList()

        for i in range(self.num_cell):
            if i == self.num_cell - 1:
                self.cells.append(Cell(modes1, modes2, modes3, width, self.skip, activation=False))
            else:
                self.cells.append(Cell(modes1, modes2, modes3, width, self.skip))
                #self.LKAs.append(LSKA(width))
            

        self.q1 = nn.Linear(self.width, self.width * 4)
        self.q2 = nn.Linear(self.width * 4, 1)

    def forward(self, x):
        # Input Shape: (B, T, C, D, H, W)
        if not self.periodic:
            x = F.pad(x, [self.padding, self.padding, self.padding, self.padding, self.padding, 
                        self.padding]) # pad the domain if input is non-periodic
        
        x = x.permute(0, 1, 3, 4, 5, 2) # (B, T, D, H, W, C)
        x = self.p(x)
        x = x.permute(0, 1, 5, 2, 3, 4) # (B, T, C, D, H, W)
        

        for i, cell in enumerate(self.cells):
            x = cell(x)
            '''
            if i < self.num_cell-1:
                lka = self.LKAs[i](x)
                x = x + lka
            '''

        if not self.periodic:
            x = x[:, :, :, self.padding:-self.padding, self.padding:-self.padding,
                 self.padding:-self.padding]

        x = x.permute(0, 1, 3, 4, 5, 2) # (B, T, D, H, W, C)
        x = self.q1(x)
        x = F.gelu(x)
        x = self.q2(x)
        x = x.permute(0, 1, 5, 2, 3, 4) # (B, T, C, D, H, W)

        return x

