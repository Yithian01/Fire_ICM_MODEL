import os
import glob
import numpy as np
from PIL import Image
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
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

        # 파일 이름에서 공통 키 추출 함수
        def core_name(f, is_mask=False):
            name = os.path.splitext(os.path.basename(f))[0]
            if is_mask and name.endswith('_CGT'):
                name = name[:-4]
            return name

        image_dict = {core_name(f): f for f in image_files}
        mask_dict = {core_name(f, is_mask=True): f for f in mask_files}

        # 공통 키만 사용
        common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

        self.img_paths = [image_dict[k] for k in common_keys]
        self.mask_paths = [mask_dict[k] for k in common_keys]

        print(f"총 이미지 파일 수: {len(image_files)}")
        print(f"총 마스크 파일 수: {len(mask_files)}")
        print(f"쌍이 맞는 이미지-마스크 개수: {len(self.img_paths)}")

        # 마스크 픽셀값 -> 클래스 인덱스 매핑
        self.mapping = {
            0: 0,      # 배경
            110: 1,
            120: 2,
            130: 3,
            140: 4,
            190: 5,
            180: 6,    # 판독불가 영역 별도 클래스
        }

    def __len__(self):
        return min(len(self.img_paths), len(self.mask_paths))

    def __getitem__(self, idx):
        # 이미지 로드
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 흑백으로 열기 (픽셀값이 클래스값)

        # 이미지 변환
        if self.transform_img:
            img = self.transform_img(img)

        # 마스크 변환 (transform_mask 내에 squeeze_mask 포함)
        if self.transform_mask:
            mask = self.transform_mask(mask)  # tensor (H,W), 클래스 인덱스가 아님

            # 마스크 픽셀값을 클래스 인덱스로 변환
            mask_np = mask.numpy()
            # 새 배열 생성
            new_mask = np.zeros_like(mask_np, dtype=np.int64)
            for k, v in self.mapping.items():
                new_mask[mask_np == k] = v
            mask = torch.from_numpy(new_mask)

        return img, mask


# 마스크 squeeze 함수
def squeeze_mask(x):
    return x.squeeze(0)


# transform_mask 전역 정의 (squeeze_mask 들여쓰기 제거)
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

    # 전처리 - 입력 이미지
    transform_img = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    # 데이터셋, 데이터로더
    train_dataset = ForestDataset(
        img_dir='./data/Training/origin/AP_IMAGE_512/',
        mask_dir='./data/Training/label/AP_512/CGT_TIF/',
        transform_img=transform_img,
        transform_mask=transform_mask
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # 모델 생성
    model = UNet(n_channels=3, n_classes=7).to(DEVICE)

    # 손실 함수와 옵티마이저
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

            if batch_idx % 10 == 0:  # 10 배치마다 loss 출력
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")


    # 학습 완료 후 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 예: 20250605_153012
    model_filename = f'unet_forest_{timestamp}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"모델이 저장되었습니다: {model_filename}")
