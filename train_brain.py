import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from model import JammerPredictorGRU

class RFDataset(Dataset):
    """Custom PyTorch Dataset for RF channel states."""
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_model(dataset_path="rf_dataset.npz", model_path="best_brain.pth", onnx_path="brain.onnx",
                hidden_dim=64, num_layers=2, batch_size=128, lr=0.001, max_epochs=20, patience=3):
    """
    Loads dataset, splits it into train/val, trains the GRU model,
    implements early stopping, and exports the final model to ONNX.
    """
    # 1. Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_dataset.py first.")
    
    with np.load(dataset_path) as data:
        X = data["X"]
        Y = data["Y"]
        
    num_samples, seq_len, num_channels = X.shape
    print(f"Loaded dataset: X shape {X.shape}, Y shape {Y.shape}")
    
    # 2. Train-Validation Split (80% train, 20% validation)
    split_idx = int(num_samples * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    train_dataset = RFDataset(X_train, Y_train)
    val_dataset = RFDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = JammerPredictorGRU(
        input_dim=num_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=num_channels
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training loop with early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    print("Beginning training...")
    for epoch in range(1, max_epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Check validation improvement for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), model_path)
            print(f"  --> Validation loss decreased. Saved model weights to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement. Early stopping counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Validation loss stalled for {patience} epochs. Stopping early!")
                break
                
    # 5. Export to ONNX
    print("\nLoading best model weights for ONNX export...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dynamic axes to allow arbitrary batch sizes during inference
    dummy_input = torch.zeros(1, seq_len, num_channels).to(device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model successfully exported to ONNX format at {onnx_path}")
    except Exception as e:
        print(f"Failed to export model to ONNX: {e}")

if __name__ == "__main__":
    train_model()
