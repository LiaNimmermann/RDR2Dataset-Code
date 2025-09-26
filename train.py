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

def train(
    data_root,
    json_path,
    log_dir="./runs/hdrcnn_hdr",
    ckpt_dir="./checkpoints",
    batch_size=4,
    lr=1e-4,
    num_epochs=50,
    device=torch.device("cuda")
):
    # Dataset and loader
    train_dataset = HDRDataset(data_root, json_path, split="train")
    val_dataset = HDRDataset(data_root, json_path, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model, optimizer
    model = SoftConvNotLearnedMaskUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # HDRLoss (perceptual + style + hole + TV)
    criterion = load("hdr", device=device)
    criterion.train()

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            ldr, hdr = batch["ldr"].to(device), batch["hdr"].to(device)

            optimizer.zero_grad()
            
                        # -------------------------
            # Dynamic mask for HDRLoss
            # -------------------------
            # Normalize LDR to [0,1] if in 0-255
            ldr_float = ldr / 255.0
            ldr_saturation_thresh = 0.95
            mask_hole_ldr = (ldr_float < ldr_saturation_thresh).float()  # 1 = valid, 0 = saturated

            # HDR-based saturation mask (e.g., pixels near max in HDR are "hole")
            hdr_max = hdr.view(hdr.shape[0], hdr.shape[1], -1).max(dim=2)[0].view(hdr.shape[0], hdr.shape[1], 1, 1)
            hdr_saturation_thresh = 0.95
            mask_hole_hdr = (hdr < hdr_max * hdr_saturation_thresh).float()  # 1 = valid, 0 = very bright

            # Combine masks: only pixels that are valid in both are safe
            mask_hole_combined = mask_hole_ldr * mask_hole_hdr

            # HDRLoss expects mask=1 for hole pixels
            mask = 1 - mask_hole_combined

            
            # Forward pass
            pred = model(ldr, input_mask=1-mask)  # input_mask: 1=valid, 0=hole


            # -------------------------
            # Compute HDR loss
            # -------------------------
            loss_dict = criterion(ldr, mask, pred, hdr)

            # Weighted sum of HDR loss components
            loss = torch.stack([
                criterion.LAMBDA_DICT[k] * loss_dict[k]
                for k in loss_dict if k in criterion.LAMBDA_DICT
            ]).sum()

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log scalar components
            for key, val in loss_dict.items():
                writer.add_scalar(f"train/{key}", val.item(), global_step)
            writer.add_scalar("train/total_loss", loss.item(), global_step)

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{batch_idx}/{len(train_loader)}] Total Loss: {loss.item():.4f}")

                # Log example images
                ldr_grid = make_grid(ldr[:4], normalize=True, scale_each=True)
                pred_grid = make_grid(pred[:4], normalize=True, scale_each=True)
                hdr_grid = make_grid(hdr[:4], normalize=True, scale_each=True)

                writer.add_image("input/ldr", ldr_grid, global_step)
                writer.add_image("output/pred_hdr", pred_grid, global_step)
                writer.add_image("target/hdr", hdr_grid, global_step)

            global_step += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_dict_sum = {k: 0.0 for k in criterion.LAMBDA_DICT.keys()}
        with torch.no_grad():
            for batch in val_loader:
                ldr, hdr = batch["ldr"].to(device), batch["hdr"].to(device)
                pred = model(ldr)
                # Dynamic mask for validation
                ldr_float = ldr / 255.0
                ldr_saturation_thresh = 0.95
                mask_hole_ldr = (ldr_float < ldr_saturation_thresh).float()

                hdr_max = hdr.view(hdr.shape[0], hdr.shape[1], -1).max(dim=2)[0].view(hdr.shape[0], hdr.shape[1], 1, 1)
                hdr_saturation_thresh = 0.95
                mask_hole_hdr = (hdr < hdr_max * hdr_saturation_thresh).float()

                mask_hole_combined = mask_hole_ldr * mask_hole_hdr
                mask = 1 - mask_hole_combined

                loss_dict = criterion(ldr, mask, pred, hdr)
                batch_loss = torch.stack([
                    criterion.LAMBDA_DICT[k] * loss_dict[k]
                    for k in loss_dict if k in criterion.LAMBDA_DICT
                ]).sum()
                val_loss += batch_loss.item()

                # accumulate per-component loss
                for k in loss_dict:
                    if k in val_loss_dict_sum:
                        val_loss_dict_sum[k] += loss_dict[k].item()

        val_loss /= len(val_loader)
        writer.add_scalar("val/total_loss", val_loss, epoch)
        for k, v in val_loss_dict_sum.items():
            writer.add_scalar(f"val/{k}", v / len(val_loader), epoch)

        print(f"Epoch {epoch+1}: train_loss={running_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        }, os.path.join(ckpt_dir, f"hdrcnn_hdr_epoch{epoch+1}.pth"))

    writer.close()


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
