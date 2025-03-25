import torch
import os
import argparse
from model import NeuralNetwork
from dataset import load_all_csvs

# Parse arguments for base folder
parser = argparse.ArgumentParser(description="Test Neural Network")
parser.add_argument("--base_folder", type=str, required=True, help="Path to dataset base folder")
parser.add_argument("--model_path", type=str, default="../models/model.pth", help="Path to trained model")
args = parser.parse_args()



# Load test data
X_train, X_val, X_test, y_train, y_val, y_test = load_all_csvs(args.base_folder, apply_pca=True)

# Load trained model
model = NeuralNetwork()
model.load_state_dict(torch.load(args.model_path))
model.eval()

X_test = X_test.view(X_test.size(0), -1)
y_test = y_test.view(-1, 1)

# Evaluate
with torch.no_grad():
    predictions = model(X_test)
    mse = torch.nn.functional.mse_loss(predictions, y_test)
    print(f"\nTest MSE Loss: {mse.item():.4f}\n")