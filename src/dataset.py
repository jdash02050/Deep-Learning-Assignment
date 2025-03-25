import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import numpy as np

pd.set_option('future.no_silent_downcasting', True)


def load_all_csvs(base_folder, apply_pca=False, n_components=3, normalize=True):
    """
    Loads all csv files from subfolders (sub01, sub02, sub03, sub05), 
    combines them into a single document, and preprocesses them.
    """
    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"the base folder was not found: {base_folder}")

    all_files = []
    subfolders = ["sub01", "sub02", "sub03", "sub05"]

    # Collect all CSV file paths
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        for file in os.listdir(subfolder_path):
            if file.endswith(".csv"):
                all_files.append(os.path.join(subfolder_path, file))

    # Read and combine all .csv files
    dataframes = []
    for f in all_files:
        df = pd.read_csv(f, header=None, names=['Index', 'A', 'B', 'C', 'Target'], index_col=0, delimiter=',')
        # Handle missing values and ensure 'Target' is numeric
        df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
        df['Target'] = df['Target'].fillna(-1)  # Handle any NaN values created during conversion
        df['C'] = pd.to_numeric(df['C'], errors='coerce')
        df['C'] = df['C'].fillna(-1)  # Handle NaN in 'C' column as well
        dataframes.append(df)

    full_data = pd.concat(dataframes, ignore_index=True)

    # Extract features and target
    X = full_data[['A', 'B', 'C']]
    y = full_data['Target']

    # Convert to numpy arrays
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop NaN values
    X = X.dropna()
    y = y.dropna()

    # Convert to numpy arrays for further processing
    X = X.to_numpy().astype(float)
    y = y.to_numpy().astype(float)

    # Apply PCA if enabled
    if apply_pca:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)

    # Normalize if enabled
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Train/Test split (90% train, 10% test)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Further split training set (20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Convert to torch tensors and ensure target is 1D
    X_train, X_val, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_val, X_test])
    y_train, y_val, y_test = map(lambda y: torch.tensor(y, dtype=torch.long).view(-1), [y_train, y_val, y_test])

    return X_train, X_val, X_test, y_train, y_val, y_test