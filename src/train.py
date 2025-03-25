import torch
import torch.optim as optim
import torch.nn as nn
from model import NeuralNetwork
from dataset import load_all_csvs
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument('--base_folder', type=str, required=True, help='Path to the base folder of the dataset')
parser.add_argument('--normalize', type=bool, default=True, help='Whether to normalize the data')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop', 'momentum'], help='Optimizer choice')
parser.add_argument('--weight_init', type=str, default='random', choices=['random', 'xavier'], help='Weight initialization method')
parser.add_argument('--l2_reg', type=float, default=0.0, help='L2 regularization (weight decay)')
args = parser.parse_args()

# Init Tensorboard writer
run_name = f"lr_{args.learning_rate}_batch_{args.batch_size}_opt_{args.weight_init}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
model_dir = os.path.join("models", run_name)
log_dir = os.path.join("runs", run_name)

os.makedirs(model_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

# Load dataset
X_train, X_val, X_test, y_train, y_val, y_test = load_all_csvs(args.base_folder, apply_pca=True, n_components=3, normalize=args.normalize)

y_train = torch.where(y_train == -1, torch.tensor(3), y_train)
y_val = torch.where(y_val == -1, torch.tensor(3), y_val)
y_test = torch.where(y_test == -1, torch.tensor(3), y_test)

# Define model
model = NeuralNetwork(weight_init=args.weight_init)

# Choose optimizer based on argument
if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_reg)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
elif args.optimizer == 'momentum':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_reg)


criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    train_loss = loss.item()
    val_loss = criterion(model(X_val), y_val).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)

    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Close Tensorboard writer
writer.close()

# Ensure models directory exists
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/model.pth")

# Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid()
plt.savefig("../models/loss_plot.png")  # Save the figure
plt.show()