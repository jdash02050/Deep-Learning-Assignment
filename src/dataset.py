import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def load_all_csvs(base_folder, apply_pca=False, n_components=3, normalize=True):
    # Assuming files are in subfolders like sub01, sub02, etc.
    all_files = []
    all_labels = []

    for subfolder in ['sub01', 'sub02', 'sub03', 'sub05']:  # Adjust according to your folders
        folder_path = os.path.join(base_folder, subfolder)
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                data = pd.read_csv(file_path)
                X = data.iloc[:, 1:4].values  # Assuming the first column is some identifier and we want columns 1, 2, 3
                y = data.iloc[:, 4].values    # Target labels in the 5th column

                # Convert # to -1
                y = np.where(y == '#', -1, y)
                y = y.astype(int)

                all_files.append(X)
                all_labels.append(y)

    X_all = np.vstack(all_files)
    y_all = np.concatenate(all_labels)

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_all)

    # Apply PCA if requested
    if apply_pca:
        pca = PCA(n_components=n_components)
        X_all = pca.fit_transform(X_all)

    # Split the data into train, validation, and test sets (80%, 10%, 10% split)
    train_size = int(0.8 * len(X_all))
    val_size = int(0.1 * len(X_all))

    X_train = torch.tensor(X_all[:train_size], dtype=torch.float32)
    y_train = torch.tensor(y_all[:train_size], dtype=torch.long)

    X_val = torch.tensor(X_all[train_size:train_size + val_size], dtype=torch.float32)
    y_val = torch.tensor(y_all[train_size:train_size + val_size], dtype=torch.long)

    X_test = torch.tensor(X_all[train_size + val_size:], dtype=torch.float32)
    y_test = torch.tensor(y_all[train_size + val_size:], dtype=torch.long)

    return X_train, X_val, X_test, y_train, y_val, y_test