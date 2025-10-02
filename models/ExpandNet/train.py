import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from tqdm import tqdm
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from RDR2_Dataloader.rdr2_hdr_dataset import HDRDataset
from util import (
    slice_gauss,
    map_range,
    cv2torch,
    random_tone_map,
    DirectoryDataset,
    str2bool,
)
from model import ExpandNet
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=12, help='Batch size.'
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=200,
        help='Checkpoint model every x epochs.',
    )
    parser.add_argument(
        '--image_log_freq',
        type=int,
        default=1,
        help='Log validation images every x epochs.',
    )
    parser.add_argument(
        '-d', '--data_root_path', default='hdr_data', help='Path to hdr data.'
    )
    parser.add_argument(
        '-s',
        '--save_path',
        default='checkpoints',
        help='Path for checkpointing.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers.',
    )
    parser.add_argument(
        '--loss_freq',
        type=int,
        default=20,
        help='Report (average) loss every x iterations.',
    )
    parser.add_argument(
        '--use_gpu', type=str2bool, default=True, help='Use GPU for training.'
    )
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of training epochs.'
    )

    return parser.parse_args()


class ExpandNetLoss(nn.Module):
    def __init__(self, loss_lambda=5):
        super(ExpandNetLoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


def transform(hdr):
    hdr = slice_gauss(hdr, crop_size=(384, 384), precision=(0.1, 1))
    hdr = cv2.resize(hdr, (256, 256))
    hdr = map_range(hdr)
    ldr = random_tone_map(hdr)
    return cv2torch(ldr).float(), cv2torch(hdr).float()


def train(opt):
    model = ExpandNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = ExpandNetLoss()

    # Datasets
    if False:
        train_dataset = HDRDataset(
            root_dir=opt.data_root_path,
            json_path=os.path.join(opt.data_root_path, 'metadata.json'),
            split='train',
            transform=transform,
            get_only_hdr=True,
        )
        val_dataset = HDRDataset(
            root_dir=opt.data_root_path,
            json_path=os.path.join(opt.data_root_path, 'metadata.json'),
            split='val',  # validation split
            transform=transform,
            get_only_hdr=True,
        )
    else:
        train_dataset = DirectoryDataset(
            data_root_path=opt.data_root_path, preprocess=transform
        )
        val_dataset = DirectoryDataset(
            data_root_path=opt.data_root_path.replace('train', 'val'),
            preprocess=transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    else:
        print(
            'WARNING: save_path already exists. '
            'Checkpoints may be overwritten'
        )

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(opt.save_path, "logs"))

    avg_loss = 0
    global_step = 0

    for epoch in tqdm(range(1, opt.epochs), desc='Training'):
        model.train()
        epoch_loss = 0

        for i, (ldr_in, hdr_target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            if opt.use_gpu:
                ldr_in = ldr_in.cuda()
                hdr_target = hdr_target.cuda()

            hdr_prediction = model(ldr_in)
            total_loss = loss_fn(hdr_prediction, hdr_target)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            avg_loss += total_loss.item()
            epoch_loss += total_loss.item()
            global_step += 1

            # log per-iteration loss
            writer.add_scalar("Loss/iter", total_loss.item(), global_step)

            if ((i + 1) % opt.loss_freq) == 0:
                rep = (
                    f'Epoch: {epoch:>5d}, '
                    f'Iter: {i+1:>6d}, '
                    f'Loss: {avg_loss/opt.loss_freq:>6.2e}'
                )
                tqdm.write(rep)
                avg_loss = 0

        # log train loss per epoch
        writer.add_scalar("Loss/train_epoch", epoch_loss / len(train_loader), epoch)

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_iter = iter(val_loader)
        sample_ldr, sample_gt = next(val_iter)
        # move to GPU if needed
        if opt.use_gpu:
            sample_ldr = sample_ldr.cuda()
            sample_gt = sample_gt.cuda()

        val_loss = 0
        with torch.no_grad():
            for ldr_val, hdr_val in val_loader:
                if opt.use_gpu:
                    ldr_val = ldr_val.cuda()
                    hdr_val = hdr_val.cuda()
                pred_val = model(ldr_val)
                val_loss += loss_fn(pred_val, hdr_val).item()

            val_loss /= len(val_loader)

            # log sample images
            if (epoch % opt.image_log_freq) == 0:
                pred_sample = model(sample_ldr).detach().cpu()
                sample_ldr = sample_ldr.detach().cpu()
                sample_gt = sample_gt.detach().cpu()

                def norm_img(x):
                    return (x - x.min()) / (x.max() - x.min() + 1e-8)

                writer.add_images("Val/Input/LDR", norm_img(sample_ldr), epoch)
                writer.add_images("Val/Output/Prediction", norm_img(pred_sample), epoch)
                writer.add_images("Val/Target/HDR", norm_img(sample_gt), epoch)


        # -------------------
        # Save checkpoint
        # -------------------
        if (epoch % opt.checkpoint_freq) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(opt.save_path, f'epoch_{epoch}.pth'),
            )

    writer.close()



"""
def train(opt):
    model = ExpandNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)
    loss = ExpandNetLoss()
    dataset = HDRDataset(
        root_dir=opt.data_root_path,
        json_path=os.path.join(opt.data_root_path, 'metadata.json'),
        split='train',
        transform=transform,
        get_only_hdr=True,
    )
    #DirectoryDataset(
    #    data_root_path=opt.data_root_path, preprocess=transform
    #)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        drop_last=True,
    )
    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    else:
        print(
            'WARNING: save_path already exists. '
            'Checkpoints may be overwritten'
        )

    avg_loss = 0
    for epoch in tqdm(range(1, 10_001), desc='Training'):
        for i, (ldr_in, hdr_target) in enumerate(
            tqdm(loader, desc=f'Epoch {epoch}')
        ):
            if opt.use_gpu:
                ldr_in = ldr_in.cuda()
                hdr_target = hdr_target.cuda()
            hdr_prediction = model(ldr_in)
            total_loss = loss(hdr_prediction, hdr_target)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            avg_loss += total_loss.item()
            if ((i + 1) % opt.loss_freq) == 0:
                rep = (
                    f'Epoch: {epoch:>5d}, '
                    f'Iter: {i+1:>6d}, '
                    f'Loss: {avg_loss/opt.loss_freq:>6.2e}'
                )
                tqdm.write(rep)
                avg_loss = 0

        if (epoch % opt.checkpoint_freq) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(opt.save_path, f'epoch_{epoch}.pth'),
            )
"""

if __name__ == '__main__':
    opt = parse_args()
    train(opt)
