import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from Networks import Action_Conditioned_FF


# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH = os.path.join("saved", "training_data.csv")
SAVE_DIR = "saved"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # early stopping


# --------------------------------------------------
# Dataset class
# --------------------------------------------------
class CollisionDataset(Dataset):
    def __init__(self, sensors, actions, labels):
        self.sensors = torch.tensor(sensors, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.sensors)

    def __getitem__(self, idx):
        return self.sensors[idx], self.actions[idx], self.labels[idx]


# --------------------------------------------------
# Load and preprocess data
# --------------------------------------------------
print(f"Loading data from {DATA_PATH} ...")
data = np.loadtxt(DATA_PATH, delimiter=",")
assert data.shape[1] == 7, f"Expected 7 columns, got {data.shape[1]}"

# 5 sensor columns, 1 action column, 1 label column
sensors = data[:, :5]
actions = data[:, 5]
labels = data[:, 6]

unique, counts = np.unique(labels, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

X_train, X_test, A_train, A_test, y_train, y_test = train_test_split(
    sensors, actions, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# Fit scaler on 7 features to match goal_seeking.py
train_dummy = np.concatenate([X_train, A_train.reshape(-1, 1), np.zeros((X_train.shape[0], 1))], axis=1)
scaler = StandardScaler().fit(train_dummy)
with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler trained on {scaler.n_features_in_} features (should be 7).")

# Use scaled sensors (5 columns) for training
X_train_scaled = scaler.transform(np.concatenate([X_train, np.zeros((X_train.shape[0], 2))], axis=1))[:, :5]
X_test_scaled  = scaler.transform(np.concatenate([X_test,  np.zeros((X_test.shape[0], 2))], axis=1))[:, :5]


# --------------------------------------------------
# Data loaders
# --------------------------------------------------
train_dataset = CollisionDataset(X_train_scaled, A_train, y_train)
test_dataset  = CollisionDataset(X_test_scaled,  A_test,  y_test)

class_counts = np.bincount(y_train.astype(int))
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = class_weights[y_train.astype(int)]

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --------------------------------------------------
# Model, loss, optimizer, scheduler
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Action_Conditioned_FF().to(device)

pos_weight_value = class_counts[0] / max(class_counts[1], 1)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
)

# --------------------------------------------------
# Training loop
# --------------------------------------------------
train_losses, val_losses, f1_scores = [], [], []
best_loss, patience_counter = float("inf"), 0

print("\nStarting training...\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for sensors_batch, actions_batch, labels_batch in train_loader:
        sensors_batch, actions_batch, labels_batch = (
            sensors_batch.to(device),
            actions_batch.to(device),
            labels_batch.to(device),
        )

        optimizer.zero_grad()
        outputs = model(sensors_batch, actions_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * sensors_batch.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for sensors_batch, actions_batch, labels_batch in test_loader:
            sensors_batch, actions_batch, labels_batch = (
                sensors_batch.to(device),
                actions_batch.to(device),
                labels_batch.to(device),
            )
            outputs = model(sensors_batch, actions_batch)
            val_loss = criterion(outputs, labels_batch)
            total_val_loss += val_loss.item() * sensors_batch.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            y_true.extend(labels_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_val_loss = total_val_loss / len(test_loader.dataset)
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch}/{EPOCHS}] - "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"F1: {f1:.3f}"
    )

    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "saved_model.pkl"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

print("\nTraining complete.")
print("Best model saved to saved/saved_model.pkl")

# --------------------------------------------------
# Plot curves
# --------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.plot(f1_scores, label="F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Metrics Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_metrics.png"))
print("Plot saved to saved/training_metrics.png")
