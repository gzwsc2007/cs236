"""
The code is based on the original ResNet implementation from torchvision.models.resnet
"""

import torch
import torch.nn as nn

DATA_LENGTH_AT_BOTTLENECK = 7 # depends on # of conv layers with >1 strides.
last_channel_size = 512

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual_shortcut=True):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                     padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.residual_shortcut = residual_shortcut
        self.downsample = None
        
        # Note that `downsample` is only used for residual computation.
        if residual_shortcut:
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out

class TransposedDecov1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0):
        super(TransposedDecov1D, self).__init__()

        # Mirror of BasicBlock1D.
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False, output_padding=output_padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class ResNetEncoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, first_channel_size=64, last_channel_size=512,
                 fc_dim=256, dropout=0.5, zero_init_residual=False):
        super(ResNetEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.last_channel_size = last_channel_size

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(feature_dim, first_channel_size, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(first_channel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            # At this point the data length is 50.
        )

        # Residual groups
        self.residual_groups = nn.Sequential(
            # Residual group 1
            BasicBlock1D(in_channels=first_channel_size,
                         out_channels=64,
                         kernel_size=3,
                         stride=1),
            BasicBlock1D(in_channels=64,
                         out_channels=64,
                         kernel_size=3,
                         stride=1),
            
            # Residual group 2
            BasicBlock1D(in_channels=64,
                         out_channels=128,
                         kernel_size=3,
                         stride=2),
            # Data length becomes 25.
            BasicBlock1D(in_channels=128,
                         out_channels=128,
                         kernel_size=3,
                         stride=1),
            
            # Residual group 3
            BasicBlock1D(in_channels=128,
                         out_channels=256,
                         kernel_size=3,
                         stride=2),
            # Data length becomes 13.
            BasicBlock1D(in_channels=256,
                         out_channels=256,
                         kernel_size=3,
                         stride=1),
            
            # Residual group 4
            BasicBlock1D(in_channels=256,
                         out_channels=last_channel_size,
                         kernel_size=3,
                         stride=2),
            # Data length becomes 7.
            BasicBlock1D(in_channels=last_channel_size,
                         out_channels=last_channel_size,
                         kernel_size=3,
                         stride=1)
        )

        # Output modules
        self.output_layer = nn.Sequential(
            nn.Linear(last_channel_size * DATA_LENGTH_AT_BOTTLENECK, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, latent_dim))

        self._initialize(zero_init_residual)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)

        shape = x.shape
        x = x.reshape((shape[0], self.last_channel_size * DATA_LENGTH_AT_BOTTLENECK))
        
        return self.output_layer(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Decoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, first_channel_size=64, last_channel_size=512,
                 fc_dim=256):
        super(Decoder, self).__init__()
        self.last_channel_size = last_channel_size

        self.fc_layer = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU(True),
            nn.Linear(fc_dim, last_channel_size * DATA_LENGTH_AT_BOTTLENECK)
        )

        self.deconv1 = nn.Sequential(
            # Opposite of residual group 4.
            BasicBlock1D(in_channels=last_channel_size,
                         out_channels=last_channel_size,
                         kernel_size=3,
                         stride=1,
                         residual_shortcut=False),
            TransposedDecov1D(in_channels=last_channel_size,
                              out_channels=256,
                              kernel_size=3,
                              stride=2),
            # Data length is 13 at this point.
        )

        self.deconv2 = nn.Sequential(
            # Opposite of residual group 3.
            BasicBlock1D(in_channels=256,
                         out_channels=256,
                         kernel_size=3,
                         stride=1,
                         residual_shortcut=False),
            TransposedDecov1D(in_channels=256,
                              out_channels=128,
                              kernel_size=3,
                              stride=2),
            # Data length is 25 at this point.
        )

        self.deconv3 = nn.Sequential(
            # Opposite of residual group 2.
            BasicBlock1D(in_channels=128,
                         out_channels=128,
                         kernel_size=3,
                         stride=1,
                         residual_shortcut=False),
            TransposedDecov1D(in_channels=128,
                              out_channels=64,
                              kernel_size=3,
                              stride=2,
                              output_padding=1),
            # Data length is 50 at this point.
        )

        self.deconv4 = nn.Sequential(
            # Opposite of residual group 1.
            BasicBlock1D(in_channels=64,
                         out_channels=64,
                         kernel_size=3,
                         stride=1,
                         residual_shortcut=False),
            TransposedDecov1D(in_channels=64,
                              out_channels=first_channel_size,
                              kernel_size=3,
                              stride=1),
            # Data length is 50 at this point.
        )

        # Opposite of the input module.
        self.output_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            # Data length is 100 at this point.
            nn.ConvTranspose1d(first_channel_size, feature_dim, kernel_size=7, stride=2, padding=3,
                               bias=False, output_padding=1)
            # Data length is 200 at this point.
        )

    def forward(self, x):
        x = self.fc_layer(x)
        x = x.reshape((x.shape[0], self.last_channel_size, DATA_LENGTH_AT_BOTTLENECK))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return self.output_block(x)

class CnnAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, first_channel_size, last_channel_size,
                 fc_dim):
        super(CnnAutoencoder, self).__init__()
        self.name = "CnnAe_feat={}_latent={}_firstChan={}_lastChan={}_fcDim={}".format(
            feature_dim, latent_dim, first_channel_size, last_channel_size, fc_dim)
        self.latent_dim = latent_dim
        self.enc = ResNetEncoder(feature_dim,
                                 latent_dim,
                                 first_channel_size,
                                 last_channel_size,
                                 fc_dim)
        self.dec = Decoder(feature_dim,
                           latent_dim,
                           first_channel_size,
                           last_channel_size,
                           fc_dim)
    
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

class VelocityRegressor(nn.Module):
    def __init__(self, cnn_encoder, fc_dims, num_outputs, dropout=0.25):
        super(VelocityRegressor, self).__init__()
        self.cnn_encoder = cnn_encoder
        
        velocity_layers = [
            nn.Linear(cnn_encoder.latent_dim, fc_dims[0]),
            nn.ReLU(True),
            nn.Dropout(dropout)
        ]
        for i in range(1, len(fc_dims)):
            velocity_layers += [
                nn.Linear(fc_dims[i-1], fc_dims[i]),
                nn.ReLU(True),
                nn.Dropout(dropout)
            ]
        velocity_layers.append(nn.Linear(fc_dims[-1], num_outputs))
        
        self.velocity_layers = nn.Sequential(*velocity_layers)

    def forward(self, x):
        x = self.cnn_encoder(x)
        return self.velocity_layers(x)
        