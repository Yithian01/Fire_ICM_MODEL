import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

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
            0: 0, 180: 0,
            110: 1, 120: 1, 130: 1,
            140: 2,
            190: 3
        }

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
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
        except Exception as e:
            print(f"⚠️ Error loading idx {idx}: {e}")
            import traceback; traceback.print_exc()
            return None, None