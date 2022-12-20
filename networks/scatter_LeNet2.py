import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class scatter_LeNet2(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3*49, 32, 5, bias=False, padding=2) # 3*81 for scattering
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(8* 4 * 4, self.rep_dim, bias=False) # 8 for scattering os 128

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        # x = F.leaky_relu(self.bn2d2(x)) #for scatterin J=4 use this line and comment next
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        # x = F.leaky_relu(self.bn2d3(x)) #for scatterin J=3 use this line and comment next
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class scatter_LeNet2_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3*49, 32, 5, bias=False, padding=2)# 3*81 for scattering
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(8 * 4 * 4, self.rep_dim, bias=False) # 8for scatterin os 128
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)# for J=4 (2*2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3*49, 5, bias=False, padding=2)# 3*81 for scattering
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        # x = F.leaky_relu(self.bn2d2(x)) #for scatterin J=4 use this line and comment next
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        # x = F.leaky_relu(self.bn2d3(x)) #for scatterin J=3 use this line and comment next
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4) # for J=4 (2*2,2,2)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        # x = F.leaky_relu(self.bn2d4(x))# for scattering J=3 use this and comment next
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.leaky_relu(self.bn2d5(x)) # for scattering J=2 use this and comment next
        # x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.leaky_relu(self.bn2d6(x))# for scattering use this and comment next
        # x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x
