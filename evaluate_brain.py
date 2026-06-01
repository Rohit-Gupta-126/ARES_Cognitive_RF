import numpy as np
import torch
import time
import os
from model import JammerPredictorGRU

def evaluate_model(dataset_path="rf_dataset.npz", model_path="best_brain.pth", threshold=0.5):
    """
    Evaluates the trained GRU model, prints Precision, Recall, F1-score,
    and runs a speed benchmark to verify the sub-5ms latency constraint.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")

    # 1. Load dataset splits (identical split logic to train_brain.py)
    with np.load(dataset_path) as data:
        X = data["X"]
        Y = data["Y"]
        
    num_samples, seq_len, num_channels = X.shape
    split_idx = int(num_samples * 0.8)
    X_val = torch.tensor(X[split_idx:], dtype=torch.float32)
    Y_val = torch.tensor(Y[split_idx:], dtype=torch.float32)
    
    print(f"Loaded evaluation split: {len(X_val)} validation samples.")

    # 2. Load model
    device = torch.device("cpu") # Benchmark on CPU to be representative of low-power edge nodes
    print(f"Evaluating on device: {device}")
    
    model = JammerPredictorGRU(
        input_dim=num_channels,
        hidden_dim=64,
        num_layers=2,
        output_dim=num_channels
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Model validation pass
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

    # Calculate metrics
    # Convert to numpy for convenience
    y_true = Y_val.numpy()
    y_pred = preds.numpy()

    # Calculate element-wise TP, FP, TN, FN
    tp = np.sum((y_true == 1.0) & (y_pred == 1.0))
    fp = np.sum((y_true == 0.0) & (y_pred == 1.0))
    fn = np.sum((y_true == 1.0) & (y_pred == 0.0))
    tn = np.sum((y_true == 0.0) & (y_pred == 0.0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate simple Accuracy (percentage of correctly classified individual channel states)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    # Calculate Frame Accuracy: does the predicted vector exactly match the true vector?
    exact_matches = np.all(y_true == y_pred, axis=1)
    frame_accuracy = np.mean(exact_matches)

    print("\n--- PERFORMANCE EVALUATION METRICS ---")
    print(f"True Positives (TP) : {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN) : {tn}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1-Score            : {f1:.4f}")
    print(f"Bit-wise Accuracy   : {accuracy*100:.2f}%")
    print(f"Frame-wise Accuracy : {frame_accuracy*100:.2f}%")
    print("--------------------------------------")

    # 4. Latency Benchmark (sub-5ms test)
    print("\nRunning inference speed benchmark...")
    num_runs = 1000
    latencies = []
    
    # Create single-sample tensor
    single_sample = torch.zeros(1, seq_len, num_channels).to(device)
    
    # Warm-up run
    with torch.no_grad():
        _ = model(single_sample)
        
    for _ in range(num_runs):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(single_sample)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # milliseconds
        
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"Average CPU Inference Latency: {avg_latency:.4f} ms")
    print(f"95th Percentile Latency       : {p95_latency:.4f} ms")
    print(f"Max Inference Latency        : {max_latency:.4f} ms")
    
    if avg_latency < 5.0:
        print("SUCCESS: Inference speed meets sub-5ms requirement!")
    else:
        print("WARNING: Inference speed exceeds sub-5ms requirement.")

    # 5. Visual Demonstration of Prediction Sequence
    print("\nVisual Sample (First 15 validation steps):")
    print("Step | Active True Jammed Channels | Predicted Jammed Channels")
    print("-" * 65)
    for i in range(15):
        true_channels = np.where(y_true[i] == 1.0)[0].tolist()
        pred_channels = np.where(y_pred[i] == 1.0)[0].tolist()
        print(f"{i:4d} | {str(true_channels):28s} | {str(pred_channels)}")

if __name__ == "__main__":
    evaluate_model()
