from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import config

class ImagesDataset(Dataset):
    def __init__(self, root_dir, test_mode=False):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.test_mode = test_mode
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        half_width = image.shape[1] // 2
        input_image = image[:, half_width:, :]
        target_image = image[:, :half_width, :]
        
        if not self.test_mode:
            augmentations = config.transform_test(image=input_image, image0=target_image)
            input_image, target_image = augmentations["image"], augmentations["image0"]
        else:
            augmentations = config.both_transform(image=input_image, image0=target_image)
            input_image, target_image = augmentations["image"], augmentations["image0"]

            input_image = config.transform_only_input(image=input_image)["image"]
            target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
    
if __name__ == "__main__":
    dataset = ImagesDataset("D:/MACHINE LEARNING/DATASETS/Pix2Pix/data/train")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()