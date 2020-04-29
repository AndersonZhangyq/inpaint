from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from utils import *
import numpy as np


def get_mask_path(image_path: Path):
    parts = list(image_path.parts)
    parts[-2] = "mask"
    parts[-1] = parts[-1].replace("_masked.png", "_rect_mask.png")
    mask_path = Path(*parts)
    if mask_path.exists():
        return mask_path
    return False


class DeepImagePriorDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.images_path = []
        self._get_images_path()
        self.img = None
        self.mask = None
        self.noise = None
        self.context_mask = None

    def _get_images_path(self):
        image_path = self.data_path / "masked"
        mask_path = self.data_path / "mask"
        if not image_path.exists():
            raise RuntimeError(
                "Image path: {} / {} not found".format(image_path, image_path.resolve())
            )
        if not mask_path.exists():
            raise RuntimeError(
                "Mask path: {} / {} not found".format(mask_path, mask_path.resolve())
            )
        for f in image_path.iterdir():
            if f.is_file():
                print("image_path:", f.resolve())
                mask_path = get_mask_path(f)
                if mask_path == False:
                    continue
                print("mask_path:", mask_path.resolve())
                self.images_path.append((f, mask_path))

    def __getitem__(self, index):
        img_path, mask_path = self.images_path[index]
        if self.img is None:
            self.img = Image.open(img_path).convert("RGB")
        if self.mask is None:
            self.mask = Image.open(mask_path).resize(self.img.size).convert("1")
        if self.noise is None:
            self.noise = get_noise(2, "noise", self.img.size)
        if self.context_mask is None:
            array_mask = np.array(self.mask, dtype=np.float32)
            row, col = np.where(array_mask == 0)
            row_1, row_2 = row[0], row[-1]
            col_1, col_2 = col[0], col[-1]
            array_context_mask = np.copy(array_mask)
            array_context_mask[
                max(row_1 - 7, 0) : min(row_2 + 7, self.mask.height),
                max(col_1 - 7, 0) : min(col_2 + 7, self.mask.width),
            ] = 5
            i = 1
            gamma = 0.99
            while row_1 <= row_2 and col_1 <= col_2:
                array_context_mask[row_1, col_1 : col_2 + 1] = gamma ** i
                array_context_mask[row_1, col_1 : col_2 + 1] = gamma ** i
                array_context_mask[row_1 : row_2 + 1, col_1] = gamma ** i
                array_context_mask[row_1 : row_2 + 1, col_2] = gamma ** i
                row_1 += 1
                row_2 -= 1
                col_1 += 1
                col_2 -= 1
                i += 1
            self.context_mask = array_context_mask
        return self.noise, ToTensor()(self.img), ToTensor()(self.mask), torch.from_numpy(self.context_mask)

    def __len__(self):
        return len(self.images_path)
