import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

class VehicleDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.tensor([0])
        return image, label

if __name__ == "__main__":
    test_dir = "data/MOT17/MOT17/train/MOT17-04-FRCNN/img1"
    if os.path.exists(test_dir):
        dataset = VehicleDataset(test_dir)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        images, labels = next(iter(dataloader))
        print("Shape batch:", images.shape)
    else:
        print("Không tìm thấy thư mục.")
