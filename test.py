import sys
import numpy as np
from PIL import Image
import torch
from utils import load_checkpoint, save_some_examples, save_image
import torch.nn as nn
import torch.optim as optim
import config
from dataset import ImagesDataset
from generator_model import Generator
from torch.utils.data import DataLoader

def main():
    print(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)

    # If there are no arguments, randomly sample images and generate them
    if len(sys.argv) <= 1:
        val_dataset = ImagesDataset(root_dir="data/val", test_mode=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

        for epoch in range(10):
            save_some_examples(gen, val_loader, epoch, folder="test", join_dim=2, mode="test")
    # Otherwise use input image
    else:
        input_image = np.array(Image.open(sys.argv[1]))[:,:,:3]
        input_image = config.transform_test(image=input_image)["image"]
        input_image = input_image[np.newaxis]
        input_image = input_image.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            y_fake = gen(input_image)
            save_image(y_fake * 0.5 + 0.5, f"result.png")
        gen.train()


if __name__ == "__main__":
    main()