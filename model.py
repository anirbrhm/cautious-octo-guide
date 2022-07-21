import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class TransposeUpSampleLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride = 1, padding = 11, upsample = 0, output_padding = 1):
        super(TransposeUpSampleLayer, self).__init__()
        self.upsample = upsample
        self.upsample_layer = torch.nn.Upsample(scale_factor = upsample)

        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value = 0)

        self.conv1d = torch.nn.Conv1d(input_channels, output_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x) 
            padded_x = self.reflection_pad(x) 
            output = self.conv1d(padded_x)
            return output
        else:
            return self.Conv1dTrans(x)

class Generator(nn.Module):
    def __init__(self, model_size = 64, ngpus = 1, num_channels = 1, latent_dim = 128, last_filter_len = 512, upsample = True):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256 * model_size)
        self.model_size = model_size

        stride = 4
        if upsample:
            stride = 1
            upsample = 4
        self.deconv_1 = TransposeUpSampleLayer(16 * model_size, 8 * model_size, 25, stride, upsample = upsample)
        self.deconv_2 = TransposeUpSampleLayer(8 * model_size, 4 * model_size, 25, stride, upsample = upsample)
        self.deconv_3 = TransposeUpSampleLayer(4 * model_size, 2 * model_size, 25, stride, upsample = upsample)
        self.deconv_4 = TransposeUpSampleLayer(2 * model_size, model_size, 25, stride, upsample = upsample)
        self.deconv_5 = TransposeUpSampleLayer(model_size, num_channels, 25, stride, upsample = upsample)

        if last_filter_len:
            self.ppfilter1 = nn.Conv1d(num_channels, num_channels, last_filter_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.fc1(x).view(-1, 16 * self.model_size, 16)
        return torch.tanh(self.deconv_5(F.relu(self.deconv_4(F.relu(self.deconv_3(F.relu(self.deconv_2(F.relu(self.deconv_1(F.relu(x)))))))))))

class PhaseShuffle(nn.Module):
    def __init__(self, n):
        super(PhaseShuffle, self).__init__()
        self.n = n

    def forward(self, x):
        if self.n == 0:
            return x

        shift_list = (torch.Tensor(x.shape[0]).random_(0, 2 * self.n + 1) - self.n).numpy().astype(int)

        shift_map = {}
        for idx, k in enumerate(shift_list):
            if k not in shift_map:
                shift_map[k] = []
            shift_map[k].append(idx)

        x_shuffled = x.clone()

        for k, idxs in shift_map.items():
            if k <= 0 :
                x_shuffled[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode = 'reflect')
            else :
                x_shuffled[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode = 'reflect')
                
        return x_shuffled

class Discriminator(nn.Module):
    def __init__(self, model_size = 64, ngpus = 1, num_channels = 1, shift_factor = 2, alpha = 0.2):
        super(Discriminator, self).__init__()

        self.alpha = alpha 
        self.model_size = model_size

        self.conv1 = nn.Conv1d(num_channels, model_size, 25, stride = 4, padding = 11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride = 4, padding = 11)
        self.conv3 = nn.Conv1d(2 * model_size, 4 * model_size, 25, stride = 4, padding = 11)
        self.conv4 = nn.Conv1d(4 * model_size, 8 * model_size, 25, stride = 4, padding = 11)
        self.conv5 = nn.Conv1d(8 * model_size, 16 * model_size, 25, stride = 4, padding = 11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)

        self.fc1 = nn.Linear(256 * model_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.ps1(F.leaky_relu(self.conv1(x), self.alpha))
        x = self.ps2(F.leaky_relu(self.conv2(x), self.alpha))
        x = self.ps3(F.leaky_relu(self.conv3(x), self.alpha))
        x = self.ps4(F.leaky_relu(self.conv4(x), self.alpha))
        x = F.leaky_relu(self.conv5(x), self.alpha)

        return self.fc1(x.view(-1, 256 * self.model_size))