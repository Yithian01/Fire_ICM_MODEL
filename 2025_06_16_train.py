import os
import glob
import numpy as np
from PIL import Image
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim

from U_NET.unet_model import UNet

# 데이터셋 정의
class ForestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform_img=None, transform_mask=None):
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        image_files = glob.glob(os.path.join(img_dir, '*.tif'))
        mask_files = glob.glob(os.path.join(mask_dir, '*.tif'))

        def core_name(f, is_mask=False):
            name = os.path.splitext(os.path.basename(f))[0]
            if is_mask and name.endswith('_FGT'):
                name = name.replace('_FGT', '')
            return name

        image_dict = {core_name(f): f for f in image_files}
        mask_dict = {core_name(f, is_mask=True): f for f in mask_files}

        common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
        self.img_paths = [image_dict[k] for k in common_keys]
        self.mask_paths = [mask_dict[k] for k in common_keys]

        print(f"총 이미지 파일 수: {len(image_files)}")
        print(f"총 마스크 파일 수: {len(mask_files)}")
        print(f"쌍이 맞는 이미지-마스크 개수: {len(self.img_paths)}")

        self.mapping = {
            0: 0, 180: 0,         # 배경 + 판독불가
            110: 1, 120: 1, 130: 1,  # 침엽수
            140: 2,               # 활엽수
            190: 3                # 기타
        }

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform_img:
            img = self.transform_img(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask_np = mask.numpy()
            new_mask = np.zeros_like(mask_np, dtype=np.int64)
            for k, v in self.mapping.items():
                new_mask[mask_np == k] = v
            mask = torch.from_numpy(new_mask)

        return img, mask

# 마스크 전처리
def squeeze_mask(x):
    return x.squeeze(0)

# 검증 함수
def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_pixels = 0
    iou_scores = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

            # IoU 계산
            for cls in range(num_classes):
                pred_inds = (preds == cls)
                target_inds = (masks == cls)
                intersection = (pred_inds & target_inds).sum().item()
                union = (pred_inds | target_inds).sum().item()
                if union > 0:
                    iou_scores[cls] += intersection / union
                    class_counts[cls] += 1

    val_loss /= len(val_loader.dataset)
    pixel_acc = total_correct / total_pixels
    mean_iou = np.mean([iou_scores[i] / class_counts[i] if class_counts[i] > 0 else 0 for i in range(num_classes)])

    return val_loss, pixel_acc, mean_iou

# 실행 시작
if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 4
    EPOCHS = 20
    LR = 1e-3
    VAL_RATIO = 0.2
    NUM_CLASSES = 4

    transform_img = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    transform_mask = T.Compose([
        T.Resize((512, 512), interpolation=Image.NEAREST),
        T.PILToTensor(),
        squeeze_mask,
    ])

    # 데이터셋 생성 및 분할
    full_dataset = ForestDataset(
        img_dir='./data/Training/origin/AP_IMAGE_512/',
        mask_dir='./data/Training/label/AP_512/FGT_TIF/',
        transform_img=transform_img,
        transform_mask=transform_mask
    )

    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"학습 세트 크기: {train_size}, 검증 세트 크기: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    # 학습 시작
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        val_loss, pixel_acc, mean_iou = validate(model, val_loader, criterion, DEVICE, NUM_CLASSES)

        print(f"📘 Epoch {epoch+1} 결과:")
        print(f"    Train Loss: {epoch_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Pixel Acc:  {pixel_acc:.4f}")
        print(f"    Mean IoU:   {mean_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = 'unet_forest_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ 새로운 최적 모델 저장됨: {best_model_path}")

    # 최종 모델 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'unet_forest_{timestamp}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"✅ 최종 모델 저장 완료: {model_filename}")
