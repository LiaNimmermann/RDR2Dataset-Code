import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from RDR2_Dataloader.rdr2_hdr_dataset import HDRDataset
from models.hdrcnn import SoftConvNotLearnedMaskUNet
from loss import load  # your HDRLoss loader

#imports for expandNet
from models.ExpandNet.util import (
    slice_gauss, 
    map_range,
    cv2torch,
    random_tone_map,
    DirectoryDataset,
    str2bool
)
from models.ExpandNet.model import ExpandNet, ExpandNetLoss
from models.ExpandNet.train import transform as expandNet_transform


def train(
    data_root,
    json_path,
    log_dir="./runs/hdrcnn_hdr",
    ckpt_dir="./checkpoints",
    batch_size=4,
    lr=7e-5,
    num_epochs=10,
    device=torch.device("cuda")
):
    
    # Dataset and loader
    train_dataset = HDRDataset(data_root, json_path, split="train", get_only_hdr=True, transform=expandNet_transform)
    val_dataset = HDRDataset(data_root, json_path, split="val")
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model, optimizer
    model = ExpandNet().to(device)
    loss = ExpandNetLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # TensorBoard writer
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            ldr, hdr = batch["hdr"].to(device)
            pred = model(ldr)
            
            l = loss(pred, hdr)
            
            optimizer.zero_grad()

            # Backpropagation
            l.backward()
            optimizer.step()

            running_loss += loss.item()

            global_step += 1

        


if __name__ == "__main__":
    train(
        data_root="C:/Test_dataset/rdr2_test_dataset_new_coco_id",
        json_path="C:/Test_dataset/rdr2_test_dataset_new_coco_id/all_captures.json",
        log_dir="./runs/hdrcnn_hdr",
        ckpt_dir="./checkpoints",
        batch_size=1,
        lr=1e-4,
        num_epochs=20
    )
