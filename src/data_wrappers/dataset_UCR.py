import sys
from pathlib import Path

from pyexpat import features

project_root = Path(__file__).resolve().parent.parent  # Goes up two levels
sys.path.append(str(project_root))

from utils import yaml_config_hook
from sktime.datasets import load_UCR_UEA_dataset
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
from sklearn.preprocessing import StandardScaler

def normalize_3d_array(X_train, X_val, X_test):

    # Reshape the 3D arrays to 2D arrays for each feature
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled

parser = argparse.ArgumentParser()
config = yaml_config_hook( project_root /"config"/"UCR_config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

args = parser.parse_args()


np.random.seed(args.seed)
print("Dataset:", args.dataset, args.dataset_)


X_train, y_train = load_UCR_UEA_dataset(name=args.dataset_, split="train", return_type="numpy3d")
X_test, y_test = load_UCR_UEA_dataset(name=args.dataset_, split="test", return_type="numpy3d")


# Encode labels to 0, 1
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Optional: further split train into train/val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

X_train, X_val, X_test = normalize_3d_array(X_train, X_val, X_test)


# Take partial features
# n = 160
# n = X_train.shape[-1]
n =  args.n_length

# Convert to PyTorch tensors
train_x = torch.tensor(X_train[:, :, :n], dtype=torch.float32)
train_y = torch.tensor(y_train, dtype=torch.long)

val_x = torch.tensor(X_val[:, :, :n], dtype=torch.float32)
val_y = torch.tensor(y_val, dtype=torch.long)

test_x = torch.tensor(X_test[:, :, :n], dtype=torch.float32)
test_y = torch.tensor(y_test, dtype=torch.long)



print(f"After split feature shape:\ntrain:{train_x.shape}\nval:{val_x.shape}\ntest:{test_x.shape}")