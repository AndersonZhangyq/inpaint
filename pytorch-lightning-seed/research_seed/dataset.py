from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from research_seed.utils import *

class DeepImagePriorDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.images_path = []
        self._get_images_path()
        self.img = None
        self.mask = None
        self.noise = None

    def _get_mask_path(self, image_path: Path):
        parts = list(image_path.parts)
        parts[-2] = "mask"
        parts[-1] = parts[-1].replace("_masked.png", "_text_mask.png")
        mask_path = Path(*parts)
        if mask_path.exists():
            return mask_path
        return False

    def _get_images_path(self):
        image_path = self.data_path/"masked"
        mask_path = self.data_path/"mask"
        if not image_path.exists():
            raise RuntimeError("Image path: {} not found".format(image_path))
        if not mask_path.exists():
            raise RuntimeError("Mask path: {} not found".format(mask_path))
        for f in image_path.iterdir():
            if f.is_file():
                mask_path = self._get_mask_path(f)
                if mask_path == False:
                    continue
                self.images_path.append((f, mask_path))
    
    def __getitem__(self, index):
        img_path, mask_path = self.images_path[index]
        if self.img is None:
            self.img = Image.open(img_path).convert('RGB')
        if self.mask is None:
            self.mask = Image.open(mask_path).convert('1')
        if self.noise is None:
            self.noise = get_noise(2, "noise", self.img.size)
        return self.noise, ToTensor()(self.img), ToTensor()(self.mask)

    def __len__(self):
        return len(self.images_path)