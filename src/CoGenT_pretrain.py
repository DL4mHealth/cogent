import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from tqdm import tqdm

from build_dataset import CustomTensorDataset
from dataset_registry import get_dataset
from modules import CoGenT_TS, load_optimizer, save_model, NT_Xent_GPU
from utils import yaml_config_hook
from utils.transformations import augmap

dataname = "UCR"


def train(epoch, args, train_loader, model, criterion, optimizer, mode="update"):
    loss_epoch = []
    epoch_loss_c = []
    epoch_loss_r = []


    for step, ((x_i, x_j), _) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        x_i = x_i.to(args.device, non_blocking=True)
        x_j = x_j.to(args.device, non_blocking=True)

        x_hat_i, x_hat_j, mask_i, mask_j, z_i, z_j = model(x_i, x_j)

        loss_r_i = torch.mean((x_hat_i - x_i) ** 2 * mask_i) / args.mask_ratio
        loss_r_j = torch.mean((x_hat_j - x_j) ** 2 * mask_j) / args.mask_ratio
        loss_c = criterion(z_i, z_j)
        loss_r = loss_r_i + loss_r_j

        loss = loss_r_i + loss_r_j + loss_c

        if mode == "update":  # only update in training, not in validation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = loss_r_i + loss_r_j + loss_c
        loss_epoch.append(loss.item())
        epoch_loss_c.append(loss_c.item())
        epoch_loss_r.append(loss_r.item())

    mean_loss = sum(loss_epoch) / len(loss_epoch)

    return mean_loss


def main(gpu, args):
    args.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Dataset:", args.dataset)
    dataset_name = get_dataset(args.dataset)

    # add val data, split from train dataset
    train_val_seg = 0.9
    train_x = dataset_name.train_x
    train_y = dataset_name.train_y

    train_data = train_x[:round(train_x.shape[0] * train_val_seg)]
    train_val = train_x[round(train_x.shape[0] * train_val_seg):]

    train_label = train_y[:round(train_x.shape[0] * train_val_seg)]
    train_val_label = train_y[round(train_x.shape[0] * train_val_seg):]

    train_dataset = CustomTensorDataset(
        data=(train_data, train_label),
        pretrain=args.pretrain,
        transform_A=augmap.get(args.augmentation)()
    )
    val_dataset = CustomTensorDataset(
        data=(train_val, train_val_label),
        pretrain=args.pretrain,
        transform_A=augmap.get(args.augmentation)()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=0,
        # pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=2,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=0,
        # pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=2,
    )

    # initialize model
    model = CoGenT_TS(
        n_channel=args.n_channel,
        n_length=args.n_length,
        emb_dim=args.projection_dim,
        patch_size=(args.n_channel, args.patch_length),
        mask_ratio=args.mask_ratio
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent_GPU(args.batch_size, args.temperature, device=args.device)

    model = model.to(args.device)

    args.current_epoch = 0
    print("Training started.")
    lowest_loss = 100

    loss_log = []
    start_time = time.time()

    for epoch in range(args.train_epochs):
        # start_time = time.time()

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch, log_epoch = train(epoch, args, train_loader, model, criterion, optimizer)

        # val loss
        loss_val_epoch, _ = train(epoch, args, val_loader, model, criterion, optimizer, mode="Not update")

        scheduler.step()

        if loss_epoch < lowest_loss:
            print('Update saved model; update lowest loss to {}'.format(loss_epoch))
            save_model(args, model, optimizer)
            lowest_loss = loss_epoch

        loss_log.append(log_epoch)

        print(
            f"Epoch [{epoch}/{args.train_epochs}]\t Loss: {loss_epoch}\t lr: {round(lr, 8)}\t Val loss:{round(loss_val_epoch, 4)}"
        )

        # print("--- %.4s seconds ---" % (time.time() - start_time))
        args.current_epoch += 1

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # save loss
    # if args.dataset == "UCR":
    #     arch_dataset_name = args.dataset + args.dataset_
    # else:
    #     arch_dataset_name = args.dataset

    # filename = 'CoGenT_' + arch_dataset_name + 'seed' + str(args.seed) + 'pretrain_loss_' \
    #            + args.augmentation + '.csv'

    loss_log = [row[0] for row in loss_log if row]
    loss_df = pd.DataFrame(loss_log)
    print(loss_df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/" + str(dataname) + "_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()  # load all the hyper-para

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    main(1, args)
