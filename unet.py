import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # Replace BatchNorm with GroupNorm
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=mid_channels),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class MultiTaskSkinModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Base segmentation model 
        self.segmentation_model = AttentionUNet(in_channels=3, out_channels=1)
        
        # Classification path using the bottleneck features
        self.classification_path = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Linear(512, 256),  # Adjust input size based on your U-Net's bottleneck
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Get segmentation output
        seg_features, bottleneck, seg_output = self.segmentation_model(x, return_features=True)
        
        # Get classification output
        class_output = self.classification_path(bottleneck)
        
        return {
            'segmentation': seg_output,
            'classification': class_output
        }

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # The concatenated input will have in_channels (from previous layer) + skip_channels (from skip connection)
            # We reduce this to out_channels through the double conv
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure x1 and x2 have same dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, intermediate_channels):
        super().__init__()
        # For gate signal (from the gating path)
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        # For skip connection (from the skip path)
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Apply convolutions
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Ensure g1 and x1 have the same spatial dimensions
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
            
        # Add and apply ReLU
        psi = self.relu(g1 + x1)
        
        # Apply 1x1 convolution and sigmoid activation
        psi = self.psi(psi)
        
        # Multiply attention weights with input feature map
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=32, dropout=0.4, bilinear=True, deep_supervision=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.dropout = dropout
        self.deep_supervision = deep_supervision
        
        # Feature sizes
        self.features = features
        
        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2, dropout=dropout)
        self.down2 = Down(features * 2, features * 4, dropout=dropout)
        self.down3 = Down(features * 4, features * 8, dropout=dropout)
        
        # Bottom layer
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor, dropout=dropout)
        
        # Define the actual channels in each layer
        # Encoder channels
        self.enc1_ch = features  # 32
        self.enc2_ch = features * 2  # 64
        self.enc3_ch = features * 4  # 128
        self.enc4_ch = features * 8  # 256
        
        # Bottleneck channels
        self.bottleneck_ch = features * 16 // factor  # 256
        
        # Decoder channels
        self.dec1_ch = features * 8 // factor  # 128
        self.dec2_ch = features * 4 // factor  # 64
        self.dec3_ch = features * 2 // factor  # 32
        self.dec4_ch = features  # 32
        
        # Attention Gates
        self.attention1 = AttentionGate(self.bottleneck_ch, self.enc4_ch, self.enc4_ch // 2)
        self.attention2 = AttentionGate(self.dec1_ch, self.enc3_ch, self.enc3_ch // 2)
        self.attention3 = AttentionGate(self.dec2_ch, self.enc2_ch, self.enc2_ch // 2)
        self.attention4 = AttentionGate(self.dec3_ch, self.enc1_ch, self.enc1_ch // 2)
        
        # Decoder
        self.up1 = Up(self.bottleneck_ch + self.enc4_ch, self.dec1_ch, bilinear, dropout=dropout)
        self.up2 = Up(self.dec1_ch + self.enc3_ch, self.dec2_ch, bilinear, dropout=dropout)
        self.up3 = Up(self.dec2_ch + self.enc2_ch, self.dec3_ch, bilinear, dropout=dropout)
        self.up4 = Up(self.dec3_ch + self.enc1_ch, self.dec4_ch, bilinear, dropout=dropout)
        
        # Output layer
        self.outc = nn.Conv2d(self.dec4_ch, out_channels, kernel_size=1)
        
        # Add deep supervision outputs
        if deep_supervision:
            self.ds1 = nn.Conv2d(self.dec1_ch, out_channels, kernel_size=1)  # 128 channels at up1 output
            self.ds2 = nn.Conv2d(self.dec2_ch, out_channels, kernel_size=1)  # 64 channels at up2 output 
            self.ds3 = nn.Conv2d(self.dec3_ch, out_channels, kernel_size=1)  # 32 channels at up3 output
            self.ds4 = nn.Conv2d(self.dec4_ch, out_channels, kernel_size=1)  # 32 channels at up4 output
        
    def forward(self, x, return_features=False):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply attention gates and decoder blocks
        # Store each decoder output in separate variables
        attn1 = self.attention1(g=x5, x=x4)
        d1 = self.up1(x5, attn1)
        
        attn2 = self.attention2(g=d1, x=x3)
        d2 = self.up2(d1, attn2)
        
        attn3 = self.attention3(g=d2, x=x2)
        d3 = self.up3(d2, attn3)
        
        attn4 = self.attention4(g=d3, x=x1)
        d4 = self.up4(d3, attn4)
        
        # Main output
        logits = self.outc(d4)
        
        if self.deep_supervision and self.training:
            # Apply deep supervision to the intermediate decoder outputs
            ds1_logits = F.interpolate(self.ds1(d1), size=logits.shape[2:], mode='bilinear', align_corners=True)
            ds2_logits = F.interpolate(self.ds2(d2), size=logits.shape[2:], mode='bilinear', align_corners=True)
            ds3_logits = F.interpolate(self.ds3(d3), size=logits.shape[2:], mode='bilinear', align_corners=True)
            ds4_logits = F.interpolate(self.ds4(d4), size=logits.shape[2:], mode='bilinear', align_corners=True)
            
            return [torch.sigmoid(logits), torch.sigmoid(ds1_logits), 
                    torch.sigmoid(ds2_logits), torch.sigmoid(ds3_logits), 
                    torch.sigmoid(ds4_logits)]
        
        if return_features:
            return logits, [x1, x2, x3, x4, x5]
        
        return torch.sigmoid(logits)