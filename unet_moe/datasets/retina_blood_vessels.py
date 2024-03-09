import io
import zipfile

from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image 


class RetinaBloodVesselDataset(Dataset):
    def __init__(self, zip_path: str):
        zip_ref = zipfile.ZipFile(zip_path)
        self.zip_ref = zip_ref
        self.train_images_files = sorted([f for f in zip_ref.namelist() if f.startswith('Data/train/image/')])
        self.train_masks_files = sorted([f for f in zip_ref.namelist() if f.startswith('Data/train/mask/')])

        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.train_images_files)
    
    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.zip_ref.read(self.train_images_files[idx])))
        mask = Image.open(io.BytesIO(self.zip_ref.read(self.train_masks_files[idx])))

        return self.transform(image), self.transform(mask)
    

if __name__ == '__main__':
    ds = RetinaBloodVesselDataset('retina_blood_vessels.zip')

    print(ds[0][0].shape)
    print(ds[0][1].shape)