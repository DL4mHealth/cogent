import argparse
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler
from torchmetrics.classification import (Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision)
from tqdm import tqdm

from build_dataset import CustomTensorDataset
from dataset_registry import get_dataset
from modules import CoGenT_TS, ViT_Classifier
from utils import yaml_config_hook

dataname = "UCR"

warnings.filterwarnings("ignore")


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/" + str(dataname) + "_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    setup_seed(args.finetune_seed)
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device:", args.device)

    # load data
    print("Dataset:", args.dataset)
    dataset_name = get_dataset(args.dataset)

    """Adjust label ratio, balanced"""
    labelled_ratio = args.labelled_ratio
    fea, lab = [], []
    for i in range(args.n_class):
        print("There are {} samples of Class {}".format((dataset_name.train_y == i).sum(), i))
        ids = dataset_name.train_y == i
        if len(ids.shape) == 1:
            aa_x = dataset_name.train_x[ids]
            aa_y = dataset_name.train_y[ids]
        else:
            aa_x = dataset_name.train_x[ids.squeeze(1)]
            aa_y = dataset_name.train_y[ids.squeeze(1)]
        n_samples = int(aa_x.shape[0] * labelled_ratio)
        aa_x_short = aa_x[:n_samples]
        aa_y_short = aa_y[:n_samples]
        fea.append(aa_x_short)
        lab.append(aa_y_short)

    fea_flat = np.concatenate(fea, axis=0)
    lab_flat = np.concatenate(lab, axis=0)
    perm = np.random.permutation(fea_flat.shape[0])

    train_x, train_y = fea_flat[perm], lab_flat[perm]

    print("For {} label ratio, {} samples use for fine-tune.".format(labelled_ratio, train_x.shape[0]))

    train_dataset = CustomTensorDataset(
        data=(train_x, train_y), pretrain=args.pretrain
    )

    # load val data
    val_dataset = CustomTensorDataset(
        data=(dataset_name.val_x, dataset_name.val_y), pretrain=args.pretrain
    )

    # load test data
    test_dataset = CustomTensorDataset(
        data=(dataset_name.test_x, dataset_name.test_y), pretrain=args.pretrain
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        args.finetune_batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        args.finetune_batch_size,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        args.finetune_batch_size,
        shuffle=True,
        drop_last=True
    )

    model = CoGenT_TS(
        n_channel=args.n_channel,
        n_length=args.n_length,
        emb_dim=args.projection_dim,
        patch_size=(args.n_channel, args.patch_length),
        mask_ratio=args.mask_ratio
    )


    Finetune_mode = "Full"  # or "Partial" or "Full"

    if args.dataset == "UCR":
        arch_dataset_name = args.dataset + args.dataset_
    else:
        arch_dataset_name = args.dataset

    if args.pretrain:
        model_fp = os.path.join(args.model_path,
                                f"Pretrained_{arch_dataset_name}_{args.lr}_{args.projection_dim}_{args.augmentation}.tar")
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        arch = arch_dataset_name
        print("With pretrain.")

        if Finetune_mode == "Full":
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=args.finetune_base_learning_rate * args.finetune_batch_size / 256,
                                          betas=(0.9, 0.999), weight_decay=args.weight_decay)
        else:
            # Freeze first block, unfreeze second block and head
            for param in model.transformer[0].parameters():
                param.requires_grad = False
            for param in model.transformer[1].parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=args.finetune_base_learning_rate * args.finetune_batch_size / 256,
                                          betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        arch = arch_dataset_name + "_no_pre"
        print("No pretrain.")

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.finetune_base_learning_rate * args.finetune_batch_size / 256,
                                      betas=(0.9, 0.999), weight_decay=args.weight_decay)

    model = ViT_Classifier(model.encoder, num_classes=args.n_class).to(args.device)

    loss_fn = torch.nn.CrossEntropyLoss()

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.train_epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    best_val_F1 = 0
    step_count = 0
    optimizer.zero_grad()

    # save results
    finetune_results_df = pd.DataFrame(columns=['Label ratio', 'epoch', 'Loss', 'Acc',
                                                'Precision', 'Recall', 'F1', 'AUC', 'PRC'],
                                       index=range(0, int((args.finetune_epochs - 1) / 1))  # 10
                                       )
    r = 0

    for epoch in range(args.finetune_epochs):
        print("--------start fine-tuning--------")
        model.train()
        loss_epoch = 0.0
        batch_count = 0

        accuracy_metric = Accuracy(task="multiclass", num_classes=args.n_class).to(args.device)
        precision_metric = Precision(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        recall_metric = Recall(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        f1_metric = F1Score(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        auc_metric = AUROC(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        prc_metric = AveragePrecision(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)

        for sample, label in tqdm(iter(train_dataloader)):
            step_count += 1

            if args.pretrain == True:  # With pretrain, Sample is a tuple as (x_i, x_j)
                sample = sample[0].to(args.device)
            else:  # No pretrain, no augmentation
                sample = sample.to(args.device)

            label = label.to(args.device)

            logits = model(sample)[0]

            predicted = logits.argmax(1)
            label = label.type(torch.LongTensor).to(args.device)
            loss = loss_fn(logits, label.squeeze(-1))
            one_hot_y = F.one_hot(label, num_classes=args.n_class)

            accuracy_metric.update(predicted, label)
            precision_metric.update(predicted, label)
            recall_metric.update(predicted, label)
            f1_metric.update(predicted, label)

            try:
                auc_metric.update(logits, label)
            except ValueError:
                pass

            prc_metric.update(logits, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_epoch += loss.item()
            batch_count += 1

        lr_scheduler.step()

        avg_train_loss = loss_epoch / batch_count
        avg_train_acc = accuracy_metric.compute().item()
        avg_train_precision = precision_metric.compute().item()
        avg_train_recall = recall_metric.compute().item()
        avg_train_F1 = f1_metric.compute().item()

        try:
            avg_train_auc = auc_metric.compute().item()
        except ValueError:
            avg_train_auc = 0.0

        avg_train_prc = prc_metric.compute().item()

        # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')
        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{args.train_epochs}]\n average Finetune Loss: {avg_train_loss}\n Finetune Accuracy: {avg_train_acc:.4f}\n"
                f"Finetune Precision: {avg_train_precision:.4f}\n Finetune Recall: {avg_train_recall:.4f}\n "
                f"Finetune F1: {avg_train_F1:.4f}\n Finetune AUC: {avg_train_auc:.4f}\n Finetune PRC: {avg_train_prc:.4f}"
            )

        model.eval()
        print("--------start validation--------")
        val_loss = 0.0
        batch_count = 0

        accuracy_metric = Accuracy(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        precision_metric = Precision(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        recall_metric = Recall(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        f1_metric = F1Score(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        auc_metric = AUROC(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
        prc_metric = AveragePrecision(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)

        with torch.no_grad():
            for sample, label in tqdm(iter(val_dataloader)):
                if args.pretrain == True:  # With pretrain, Sample is a tuple as (x_i, x_j)
                    sample = sample[0].to(args.device)
                else:  # No pretrain, no augmentation
                    sample = sample.to(args.device)
                # sample = x_i.to(args.device)
                label = label.to(args.device)

                logits = model(sample)[0]

                predicted = logits.argmax(1)
                label = label.type(torch.LongTensor).to(args.device)

                loss = loss_fn(logits, label.squeeze(-1))
                one_hot_y = F.one_hot(label, num_classes=args.n_class)

                accuracy_metric.update(predicted, label)
                precision_metric.update(predicted, label)
                recall_metric.update(predicted, label)
                f1_metric.update(predicted, label)

                try:
                    auc_metric.update(logits, label)
                except ValueError:
                    pass

                prc_metric.update(logits, label)

                val_loss += loss.item()
                batch_count += 1

            avg_val_loss = val_loss / batch_count
            avg_val_acc = accuracy_metric.compute().item()
            avg_val_precision = precision_metric.compute().item()
            avg_val_recall = recall_metric.compute().item()
            avg_val_F1 = f1_metric.compute().item()
            try:
                avg_val_auc = auc_metric.compute().item()
            except ValueError:
                avg_val_auc = 0.0
            avg_val_prc = prc_metric.compute().item()

            if epoch % 1 == 0:
                print(
                    f"Epoch [{epoch}/{args.finetune_epochs}]\n Average validation Loss: {avg_val_loss}\n Validation Accuracy: {avg_val_acc:.4f}\n"
                    f"Validation Precision: {avg_val_precision:.4f}\n Validation Recall: {avg_val_recall:.4f}\n "
                    f"Validation F1: {avg_val_F1:.4f}\n Validation AUC: {avg_val_auc:.4f}\n Validation PRC: {avg_val_prc:.4f}"
                )

        # use F1 to select best model
        if avg_val_F1 > best_val_F1:
            best_val_F1 = avg_val_F1
            print(f'saving best model with F1 {best_val_F1} at {epoch} epoch!')
            FT_model_path = 'save/finetune/' + arch_dataset_name + str(args.labelled_ratio) + '.pt'
            torch.save(model, FT_model_path)

        best_test_F1 = 0.0

        if epoch % 1 == 0:
            print("TEST-epoch {}--------start testing-------- ".format(epoch))

            model.eval()
            test_loss = 0.0
            batch_count = 0

            all_preds = []
            all_labels = []

            accuracy_metric = Accuracy(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
            precision_metric = Precision(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
            recall_metric = Recall(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
            f1_metric = F1Score(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)

            auc_metric = AUROC(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)
            prc_metric = AveragePrecision(task="multiclass", num_classes=args.n_class, average='macro').to(args.device)

            # for visual
            all_test_representations = []
            all_test_labels = []

            with torch.no_grad():
                for sample, label in tqdm(iter(test_dataloader)):
                    if args.pretrain == True:  # With pretrain, Sample is a tuple as (x_i, x_j)
                        sample = sample[0].to(args.device)
                    else:  # No pretrain, no augmentation
                        sample = sample.to(args.device)
                    # sample = x_i.to(args.device)
                    label = label.to(args.device)

                    logits = model(sample)[0]

                    test_features = model(sample)[1]
                    all_test_representations.append(test_features.detach())
                    all_test_labels.append(label)

                    predicted = logits.argmax(1)
                    label = label.type(torch.LongTensor).to(args.device)

                    loss = loss_fn(logits, label.squeeze(-1))
                    one_hot_y = F.one_hot(label, num_classes=args.n_class)

                    accuracy_metric.update(predicted, label)
                    precision_metric.update(predicted, label)
                    recall_metric.update(predicted, label)
                    f1_metric.update(predicted, label)

                    try:
                        auc_metric.update(logits, label)
                    except ValueError:
                        pass

                    prc_metric.update(logits, label)

                    all_preds.append(predicted.cpu())
                    all_labels.append(label.cpu())

                    test_loss += loss.item()
                    batch_count += 1

                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)

                avg_test_loss = test_loss / batch_count
                avg_test_acc = accuracy_metric.compute().item()
                avg_test_precision = precision_metric.compute().item()
                avg_test_recall = recall_metric.compute().item()
                avg_test_F1 = f1_metric.compute().item()
                try:
                    avg_test_auc = auc_metric.compute().item()
                except ValueError:
                    avg_test_auc = 0.0
                avg_test_prc = prc_metric.compute().item()

                print("Confusion Matrix:")
                print(confusion_matrix(all_labels.numpy(), all_preds.numpy()))


                print(
                    f"Testing: \n Average test Loss: {avg_test_loss}\n Test Accuracy: {avg_test_acc:.4f}\n"
                    f"Test Precision: {avg_test_precision:.4f}\n Test Recall: {avg_test_recall:.4f}\n "
                    f"Test F1: {avg_test_F1:.4f}\n Test AUC: {avg_test_auc:.4f}\n Test PRC: {avg_test_prc:.4f}"
                )

                # for results
                finetune_results_df.loc[r] = pd.Series({'Label ratio': args.labelled_ratio, 'epoch': epoch,
                                                        'Loss': avg_test_loss, 'Acc': avg_test_acc,
                                                        'Precision': avg_test_precision,
                                                        'Recall': avg_test_recall,
                                                        'F1': avg_test_F1, 'AUC': avg_test_auc,
                                                        'PRC': avg_test_prc}
                                                       )
                r = r + 1

                print("Pretrain: {}; Label ratio: {}".format(args.pretrain, args.labelled_ratio))

print(finetune_results_df)


best_F1 = finetune_results_df['F1'].max()
print("Best F1: ", best_F1)

