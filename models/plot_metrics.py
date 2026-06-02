import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plots():
    # 1. Define data from the training run
    epochs = np.arange(1, 21)
    
    # Train and validation losses from the actual final run
    train_losses = [
        0.245073, 0.140415, 0.106819, 0.093804, 0.087013, 
        0.081449, 0.076852, 0.073517, 0.071153, 0.069215, 
        0.067960, 0.066851, 0.065980, 0.065320, 0.064727, 
        0.064312, 0.063867, 0.063465, 0.063147, 0.062867
    ]
    
    val_losses = [
        0.180301, 0.119964, 0.098029, 0.089989, 0.084204, 
        0.078856, 0.075100, 0.072236, 0.070334, 0.068613, 
        0.067686, 0.066648, 0.065911, 0.065481, 0.064671, 
        0.064573, 0.064179, 0.064036, 0.064183, 0.063480
    ]
    
    # Model evaluation metrics
    metrics = ["Bit-wise Accuracy", "F1-Score", "Precision", "Recall", "Frame Accuracy"]
    metric_values = [98.56, 88.99, 94.79, 83.86, 77.77] # in %
    
    # 2. Setup dark theme styles
    plt.rcParams['figure.facecolor'] = '#0f172a' # Dark Slate/Blue (next.js background)
    plt.rcParams['axes.facecolor'] = '#0b0f19' # Deeper dark
    plt.rcParams['text.color'] = '#f8fafc'
    plt.rcParams['axes.labelcolor'] = '#cbd5e1'
    plt.rcParams['xtick.color'] = '#94a3b8'
    plt.rcParams['ytick.color'] = '#94a3b8'
    plt.rcParams['axes.edgecolor'] = '#334155'
    plt.rcParams['grid.color'] = '#1e293b'
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 3. Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle("A.R.E.S. Cognitive GRU Model Performance", fontsize=16, fontweight='bold', color='#22d3ee', y=0.96)
    
    # Plot 1: Loss Convergence Curves
    ax1.plot(epochs, train_losses, label="Train Loss", color='#06b6d4', linewidth=2.5, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, label="Validation Loss", color='#ec4899', linewidth=2.5, marker='x', markersize=5)
    ax1.set_title("Loss Convergence (BCEWithLogitsLoss)", fontsize=13, fontweight='bold', color='#cbd5e1', pad=15)
    ax1.set_xlabel("Epochs", fontsize=11, labelpad=8)
    ax1.set_ylabel("Loss Value", fontsize=11, labelpad=8)
    ax1.set_xticks(range(2, 21, 2))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10)
    
    # Plot 2: Horizontal Bar Chart of Metrics
    colors = ['#10b981', '#6366f1', '#06b6d4', '#ec4899', '#f59e0b']
    bars = ax2.barh(metrics, metric_values, color=colors, height=0.55, edgecolor='#334155', alpha=0.85)
    ax2.set_title("Model Evaluation Metrics", fontsize=13, fontweight='bold', color='#cbd5e1', pad=15)
    ax2.set_xlabel("Percentage (%)", fontsize=11, labelpad=8)
    ax2.set_xlim(0, 110)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                 va='center', ha='left', fontsize=10.5, fontweight='bold', color='#f8fafc')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    # Save file
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_performance.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Performance plots successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_plots()
