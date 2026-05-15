import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# ============ 数据集类 ============
class SpotSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=512):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像和mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)  # 灰度图
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # mask转为二值（斑点=1，背景=0）
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        return image, mask

# ============ 数据增强（处理多分辨率） ============
def get_training_augmentation(target_size=512):
    return A.Compose([
        # 统一resize到固定尺寸
        A.LongestMaxSize(max_size=target_size, p=1.0),
        A.PadIfNeeded(min_height=target_size, min_width=target_size, 
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        
        # 针对斑点检测的增强
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # 颜色增强（应对机器人脸和人脸的差异）
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        
        # 模糊和噪声（增强鲁棒性）
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_augmentation(target_size=512):
    return A.Compose([
        A.LongestMaxSize(max_size=target_size, p=1.0),
        A.PadIfNeeded(min_height=target_size, min_width=target_size, 
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============ 模型选择 ============
def create_model(model_type='unet', encoder='efficientnet-b1'):
    """
    针对斑点检测的轻量模型
    """
    if model_type == 'unet':
        model = smp.Unet(
            encoder_name=encoder,        # efficientnet-b1, mobilenet_v2, resnet34
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,                   # 二分类：斑点 vs 背景
            activation=None              # 训练时用BCE，不需要sigmoid
        )
    elif model_type == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    else:  # FPN - 更轻量
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    
    return model

# ============ Loss函数（针对小目标优化） ============
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
    
    def forward(self, pred, target):
        return self.alpha * self.dice(pred, target) + (1 - self.alpha) * self.bce(pred, target)

# ============ 训练函数 ============
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ============ 验证函数 ============
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ============ 主训练流程 ============
if __name__ == '__main__':
    # 配置
    TARGET_SIZE = 512  # 统一resize到这个尺寸
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备数据路径（你需要替换这部分）
    train_image_paths = ['path/to/train/images/*.jpg']  # 你的图像路径列表
    train_mask_paths = ['path/to/train/masks/*.png']    # 对应的mask路径
    val_image_paths = ['path/to/val/images/*.jpg']
    val_mask_paths = ['path/to/val/masks/*.png']
    
    # 创建数据集
    train_dataset = SpotSegmentationDataset(
        train_image_paths, train_mask_paths,
        transform=get_training_augmentation(TARGET_SIZE)
    )
    val_dataset = SpotSegmentationDataset(
        val_image_paths, val_mask_paths,
        transform=get_validation_augmentation(TARGET_SIZE)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 创建模型（三种轻量选择）
    # model = create_model('unet', 'mobilenet_v2')        # 最轻量
    model = create_model('unet', 'efficientnet-b1')     # 推荐
    # model = create_model('unetplusplus', 'resnet34')   # 性能更好
    
    model = model.to(device)
    
    # 优化器和损失
    criterion = CombinedLoss(alpha=0.7)  # Dice权重更高，适合小目标
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_spot_model.pth')
            print(f'  → Model saved!')