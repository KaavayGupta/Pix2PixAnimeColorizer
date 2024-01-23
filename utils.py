import torch
import config
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder, join_dim = 2, mode="train"):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5 # remove normalization
        if mode == "train":
            save_image(torch.cat((x * 0.5 + 0.5, y_fake), dim=join_dim), folder + f"/gen_{epoch}.png")
            if epoch == 1:
                save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
        else:
            save_image(torch.cat((x * 0.5 + 0.5, y * 0.5 + 0.5, y_fake), dim=join_dim), folder + f"/result_{epoch}.png")
            
    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr