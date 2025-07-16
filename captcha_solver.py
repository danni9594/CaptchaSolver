import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import string

# Define the 36-class character set
CHARS = string.ascii_uppercase + string.digits

def trim_fixed(img):
    """
    Trims a fixed number of pixels from all sides of the image.
    - Horizontal: Trims 5px from left, 11px from right.
    - Vertical: Trims 11px from top, 9px from bottom.
    """
    original_width, original_height = img.size
    left = 5
    upper = 11
    right = original_width - 11
    lower = original_height - 9
    
    # Bbox is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
    box = (left, upper, right, lower)
    return img.crop(box)

# Dataset class for loading captcha images
class CaptchaDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, exclude_indices=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []
        
        # Get all available image files
        for i in range(25):  # input00.jpg to input24.jpg
            if exclude_indices and i in exclude_indices:
                continue
                
            img_path = os.path.join(image_dir, f"input{i:02d}.jpg")
            label_path = os.path.join(label_dir, f"output{i:02d}.txt")
            
            if os.path.exists(img_path) and os.path.exists(label_path):
                # Read the label
                with open(label_path, 'r') as f:
                    label = f.read().strip()
                
                if len(label) == 5:  # Ensure we have 5 characters
                    # Create 5 character samples from each captcha
                    for char_idx in range(5):
                        self.samples.append((img_path, label[char_idx], char_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, char_label, char_idx = self.samples[idx]
        
        # Load the full captcha image
        image = Image.open(img_path).convert("RGB")
        
        # Apply the fixed trim before character extraction
        image = trim_fixed(image)
        
        # Extract the character segment
        char_width = image.width // 5
        char_img = image.crop((char_idx * char_width, 0, (char_idx + 1) * char_width, image.height))
        
        # Convert to grayscale and resize
        char_img = char_img.convert("L")
        char_img = char_img.resize((32, 32))
        
        if self.transform:
            char_img = self.transform(char_img)
        else:
            # Default transform: convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            char_img = transform(char_img)
        
        # Convert character to index
        label_idx = CHARS.index(char_label)
        
        return char_img, label_idx

# Residual Block for ResNet-style architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)  # Skip connection
        out = F.relu(out)
        return out

# CNN model with ResNet-style architecture
class CaptchaCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CaptchaCNN, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Inference class
class Captcha(object):
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CaptchaCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # Minimal transform, as preprocessing is done manually
        self.transform = transforms.ToTensor()

    def __call__(self, im_path, save_path):
        image = Image.open(im_path).convert("RGB")
        # Apply the standard trim
        image = trim_fixed(image)
        
        char_width = image.width // 5
        predicted = ""
        for i in range(5):
            # Extract character segment
            char_img = image.crop((i * char_width, 0, (i + 1) * char_width, image.height))
            
            # Manual preprocessing to match train/test
            char_img = char_img.convert("L")
            char_img = char_img.resize((32, 32), Image.LANCZOS) # Use LANCZOS for older Pillow
            
            # Convert to tensor
            char_tensor = self.transform(char_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(char_tensor)
                pred_idx = output.argmax(dim=1).item()
                predicted += CHARS[pred_idx]

        with open(save_path, 'w') as f:
            f.write(predicted + "\n")
