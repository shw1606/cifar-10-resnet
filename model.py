import torch.nn as nn
import torch.nn.functional as F

class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()
        # self.pooling = nn.MaxPool2d(1, stride=stride)
        # self.add_channels = out_channels - in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # out = F.pad(x, (0,0,0,0,0,self.add_channels))
        # out = self.pooling(out)
        out = self.conv(x)
        out = self.bn(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
        out += shortcut
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10):
        super(Resnet, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer_2n = self.get_layers(block, in_channels=16, out_channels=16, stride=1)
        self.layer_4n = self.get_layers(block, in_channels=16, out_channels=32, stride=2)
        self.layer_6n = self.get_layers(block, in_channels=32, out_channels=64, stride=2)

        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc_out = nn.Linear(64, num_classes)

        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
          elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def get_layers(self, block, in_channels, out_channels, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False
        
        layers_list = nn.ModuleList(
            [block(in_channels, out_channels, stride, down_sample)]
        )

        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer_2n(x)
        x = self.layer_4n(x)
        x = self.layer_6n(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x
    

def resnet():
    block = ResidualBlock
    model = Resnet(5, block)
    return model