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
            if is_mask and name.endswith('_CGT'):
                name = name[:-4]
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
            0: 0, 180: 0,  # 배경 + 판독불가
            110: 1, 120: 1, 130: 1,  # 침엽수
            140: 2,  # 활엽수
            190: 3   # 기타
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

transform_mask = T.Compose([
    T.Resize((512, 512), interpolation=Image.NEAREST),
    T.PILToTensor(),
    squeeze_mask,
])

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 4
    EPOCHS = 20
    LR = 1e-3
    VAL_RATIO = 0.2  # 🔄 검증 비율

    transform_img = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    # 🔄 전체 데이터셋 생성
    full_dataset = ForestDataset(
        img_dir='./data/Training/origin/AP_IMAGE_512/',
        mask_dir='./data/Training/label/AP_512/CGT_TIF/',
        transform_img=transform_img,
        transform_mask=transform_mask
    )

    # 🔄 학습/검증 데이터셋 분할
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"학습 세트 크기: {len(train_dataset)}, 검증 세트 크기: {len(val_dataset)}")

    # 🔄 데이터로더 구성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    model = UNet(n_channels=3, n_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 학습 루프
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
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'unet_forest_{timestamp}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"모델이 저장되었습니다: {model_filename}")
