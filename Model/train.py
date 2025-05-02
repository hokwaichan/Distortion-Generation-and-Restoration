import os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torchvision.models import ResNet18_Weights

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

distorted_dir = " "
uv_map_dir    = " "
output_dir    = " "
save_path     = " "

batch_size    = 2
num_epochs    = 200
lr            = 0.0001

os.makedirs(output_dir, exist_ok=True)

# Dataset
class UVDataset(Dataset):
    def __init__(self, distorted_dir, uv_dir, transform=None):
        self.distorted_images = sorted(glob.glob(distorted_dir + '/*.png'))
        self.uv_maps          = sorted(glob.glob(uv_dir + '/*.npy'))
        self.transform        = transform

    def __getitem__(self, idx):
        distorted = Image.open(self.distorted_images[idx]).convert('RGB')
        uv        = np.load(self.uv_maps[idx])

        if self.transform:
            distorted = self.transform(distorted)

        uv = torch.from_numpy(uv).permute(2, 0, 1).float()
        uv = F.interpolate(uv.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)

        return distorted, uv

    def __len__(self):
        return len(self.distorted_images)

# Transforms and Dataloader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset    = UVDataset(distorted_dir, uv_map_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model Class
class UVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.decoder(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x

# Loss, Optimizer, and PSNR Function
model     = UVNet().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def uv_psnr(pred, target):
    mse  = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    for distorted_imgs, uv_maps in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        distorted_imgs, uv_maps = distorted_imgs.to(device), uv_maps.to(device)

        optimizer.zero_grad()
        outputs = model(distorted_imgs)
        loss    = criterion(outputs, uv_maps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_psnr += uv_psnr(outputs, uv_maps)

    avg_loss = running_loss / len(dataloader)
    avg_psnr = running_psnr / len(dataloader)

    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB')

# Model Save
torch.save(model.state_dict(), save_path)
print(f'Model saved to {save_path}')

# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    total_psnr = 0.0
    total_l1   = 0.0
    total_mse  = 0.0
    count      = 0

    with torch.no_grad():
        for idx, (distorted_imgs, uv_maps) in tqdm(enumerate(dataloader), desc="Evaluating"):
            distorted_imgs, uv_maps = distorted_imgs.to(device), uv_maps.to(device)
            outputs = model(distorted_imgs)

            l1_loss = F.l1_loss(outputs, uv_maps)
            mse_loss = F.mse_loss(outputs, uv_maps)
            psnr = uv_psnr(outputs, uv_maps)

            np.save(os.path.join(output_dir, f'generated_uv_map_{idx+1}.npy'), outputs.cpu().numpy())

            uv_map_np = outputs.cpu().squeeze(0).permute(1, 2, 0).numpy() 
            np.save(os.path.join(output_dir, f'generated_uv_map_{idx+1}.npy'), uv_map_np)

            uv_x = uv_map_np[..., 0]
            uv_y = uv_map_np[..., 1]

            uv_x_img = Image.fromarray(np.uint8(255 * (uv_x - uv_x.min()) / (uv_x.ptp() + 1e-8)))
            uv_y_img = Image.fromarray(np.uint8(255 * (uv_y - uv_y.min()) / (uv_y.ptp() + 1e-8)))

            uv_x_img.save(os.path.join(output_dir, f'generated_uv_map_{idx+1}_uv_x.png'))
            uv_y_img.save(os.path.join(output_dir, f'generated_uv_map_{idx+1}_uv_y.png'))
        
            total_l1   += l1_loss.item()
            total_mse  += mse_loss.item()
            total_psnr += psnr
            count      += 1

    avg_l1   = total_l1 / count
    avg_mse  = total_mse / count
    avg_psnr = total_psnr / count

    print(f"\nEvaluation Results:")
    print(f"  Mean L1 Loss (MAE): {avg_l1:.6f}")
    print(f"  Mean MSE: {avg_mse:.6f}")
    print(f"  Mean PSNR: {avg_psnr:.2f} dB")

evaluate_model(model, dataloader)
