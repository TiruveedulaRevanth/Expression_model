import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.utils import resample

# Paths
DATA_PATH = "C:\\Users\\revan\\Desktop\\face-mesh-react\\JoyVerseDataSet_Filled.xlsx"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Check CUDA
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))

# Check for NaN/Inf in Data
def check_data(data, name="Data"):
    print(f"{name} - NaN:", np.any(np.isnan(data)), "Inf:", np.any(np.isinf(data)))
    print(f"{name} - Min:", np.nanmin(data), "Max:", np.nanmax(data))

# Load and Preprocess Data
try:
    df = pd.read_excel(DATA_PATH)
    print(f"Dataset loaded: {len(df)} rows")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

df = df[df["Expression"].notna()]  # Remove rows with NaN in Expression
X = df.drop(columns=["Expression", "FileName"]).fillna(df.drop(columns=["Expression", "FileName"]).mean()).values.astype(np.float32)
y = df["Expression"]

# Strict class balancing
df_balanced = pd.DataFrame()
for expr in df["Expression"].unique():
    df_expr = df[df["Expression"] == expr]
    df_expr_resampled = resample(df_expr, n_samples=50, random_state=42, replace=True)
    df_balanced = pd.concat([df_balanced, df_expr_resampled])
X = df_balanced.drop(columns=["Expression", "FileName"]).values.astype(np.float32)
y = df_balanced["Expression"]

# Check raw data
check_data(X, "Raw X")
print("Class distribution:", pd.Series(y).value_counts())

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Dataset
class ExpressionDataset(Dataset):
    def __init__(self, X, y):
        assert X.shape[1] == 1404, f"Expected 1404 features, got {X.shape[1]}"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Enhanced MLP Model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=1404, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Training Function
def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs=100, patience=15):
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at epoch {epoch+1}")
                return False
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
                correct += (preds.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = 100 * correct / total
        val_loss /= len(val_loader)
        print(f"Validation Accuracy: {val_acc:.2f}% | Validation Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        # Early Stopping
        if val_acc > best_val_acc + 0.5:  # Significant improvement
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    return True

# Data Split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)

# Datasets and Loaders
train_ds = ExpressionDataset(X_train, y_train)
val_ds = ExpressionDataset(X_val, y_val)
test_ds = ExpressionDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleClassifier(num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.CrossEntropyLoss()

# Train
success = train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device)
if not success:
    print("Training failed due to NaN/Inf loss")
    exit(1)

# Test Evaluation
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), weights_only=True))
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)
        all_preds.extend(preds.argmax(1).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("Classes:", le.classes_)

# Save Final Model
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "expression_classifier.pt"))
print("✅ Model, scaler, and label encoder saved.")