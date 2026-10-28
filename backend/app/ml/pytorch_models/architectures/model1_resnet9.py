import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv2d -> BatchNorm2d -> ReLU
    Optionally includes MaxPool2d.
    """
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """
    Residual Block containing two ConvBlocks (without pooling).
    Includes a skip connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.final_relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.final_relu(out)
        return out

class ResNet9Model1(nn.Module):
    """
    ResNet-9 architecture for Leaf vs Non-Leaf classification.
    Input: [B, 3, 224, 224]
    Output: [B, 2] raw logits.
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        # Block 1
        self.block1 = ConvBlock(in_channels, 64, pool=False)
        
        # Block 2
        self.block2 = ConvBlock(64, 128, pool=True)
        
        # Residual Block 1
        self.res1 = ResidualBlock(128)
        
        # Block 3
        self.block3 = ConvBlock(128, 256, pool=True)
        
        # Block 4
        self.block4 = ConvBlock(256, 512, pool=True)
        
        # Residual Block 2
        self.res2 = ResidualBlock(512)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He initialization for Conv layers with ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Sensible init for BatchNorm
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Sensible init for Linear
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.res1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.res2(x)
        x = self.classifier(x)
        return x

def create_model1_resnet9():
    """
    Returns a fresh randomly initialized ResNet-9 model.
    """
    return ResNet9Model1(in_channels=3, num_classes=2)
