import numpy as np
import random
import os
from simulation.ether import RFEther
from simulation.jammer import SweepJammer, BarrageJammer, FollowerJammer, RandomJammer

def generate_simulation_data(num_steps=100000, seq_len=10, output_path="data/rf_dataset.npz"):
    """
    Runs simulations using different EW jammers and generates a dataset
    of past jammed states (X) and next jammed states (Y).
    """
    print(f"Starting dataset generation for {num_steps} total steps...")
    
    # 32 channels
    num_channels = 32
    ether = RFEther(num_channels=num_channels)
    
    # Define jammers to use in the simulation with their relative weights
    # We will run sub-simulations for each jammer type to gather diverse patterns
    jammer_configs = [
        {"type": "sweep", "weight": 0.35, "jammer": SweepJammer(num_channels, step_size=1, jam_width=1, randomness_prob=0.15)},
        {"type": "sweep_fast", "weight": 0.15, "jammer": SweepJammer(num_channels, step_size=2, jam_width=2, randomness_prob=0.1)},
        {"type": "barrage", "weight": 0.25, "jammer": BarrageJammer(num_channels, block_size=5, shift_interval=25)},
        {"type": "follower", "weight": 0.15, "jammer": FollowerJammer(num_channels)},
        {"type": "random", "weight": 0.10, "jammer": RandomJammer(num_channels, num_to_jam=2)}
    ]
    
    # Initialize buffers
    all_jammed_vectors = []
    
    # Transmitter hopping behavior simulation for the Follower jammer to sense
    # Transmitter will do a mix of random hopping, sequential hopping, and static dwelling
    tx_channel = random.randint(0, num_channels - 1)
    tx_mode = "random"
    
    for config in jammer_configs:
        jammer = config["jammer"]
        steps_for_jammer = int(num_steps * config["weight"])
        print(f"Simulating {config['type']} jammer for {steps_for_jammer} steps...")
        
        # Reset environment and jammer
        ether.reset()
        jammer.reset()
        tx_history = []
        
        for step in range(steps_for_jammer):
            # Simulate transmitter decision
            if step % 100 == 0:
                tx_mode = random.choice(["random", "sequential", "static"])
            
            if tx_mode == "random":
                tx_channel = random.randint(0, num_channels - 1)
            elif tx_mode == "sequential":
                tx_channel = (tx_channel + 1) % num_channels
            # if static, tx_channel remains the same
            
            tx_history.append(tx_channel)
            # Keep tx_history size bounded to prevent infinite growth
            if len(tx_history) > 10:
                tx_history.pop(0)
                
            # Get jammer decision
            jam_channels = jammer.get_jammed_channels(step, tx_history)
            
            # Step the simulation
            result = ether.step(tx_channel, jam_channels)
            
            # Record the binary jammed vector (1.0 if channel is jammed, 0.0 otherwise)
            all_jammed_vectors.append(result['jammed_vector'])
            
    all_jammed_vectors = np.array(all_jammed_vectors)
    total_frames = len(all_jammed_vectors)
    print(f"Simulation completed. Total frames logged: {total_frames}")
    
    # Construct sequential dataset
    # X shape: [NumSamples, seq_len, 32]
    # Y shape: [NumSamples, 32]
    num_samples = total_frames - seq_len
    
    X = np.zeros((num_samples, seq_len, num_channels), dtype=np.float32)
    Y = np.zeros((num_samples, num_channels), dtype=np.float32)
    
    for i in range(num_samples):
        X[i] = all_jammed_vectors[i : i + seq_len]
        Y[i] = all_jammed_vectors[i + seq_len]
        
    # Shuffle dataset for uniform distribution in splits
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
        
    print(f"Dataset compiled and shuffled:")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    # Save to npz file
    np.savez_compressed(output_path, X=X, Y=Y)
    print(f"Dataset successfully saved to {output_path}")
    
    return X, Y

if __name__ == "__main__":
    generate_simulation_data(num_steps=100000, seq_len=10, output_path="data/rf_dataset.npz")
