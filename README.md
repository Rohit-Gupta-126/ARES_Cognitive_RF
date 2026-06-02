# Project A.R.E.S. (Autonomous Radio Evasion System)

Project A.R.E.S. is an intelligent radio system designed to protect drone communication and telemetry links from targeted signal jamming (Electronic Warfare). 

It simulates a 32-channel radio spectrum where an AI "brain" predicts where jammers will strike next and automatically hops to clear frequencies. To prevent smart jammers from guessing our next move, the system combines AI predictions with a hardware security device called the **ZenTropy Key (ZEK)** to select channels in a completely unpredictable, secure way.

---

## 💡 The Core Concept: Hybrid AI-Cryptographic Evasion

Instead of letting the AI make the final channel decision (which might follow a predictable pattern that a smart tracker could exploit), A.R.E.S. uses a hybrid pipeline:
1. **Identify the Safe Zone**: The AI evaluates the spectrum and ranks the channels, selecting the top-5 safest candidates (the "Safe Zone").
2. **Cryptographic Selection**: The ZenTropy Key (using physical noise or secure software fallback) randomly selects the final channel from this safe zone.

A.R.E.S. supports two different AI brains:
* **The Pattern Predictor (GRU)**: A pattern-matching model that analyzes history to predict the probability of each channel being jammed.
* **The Self-Learning Agent (DQN)**: A reinforcement learning model that learns through trial-for-error, scoring points for successful transmissions and receiving penalties for collisions.

---

## 📂 Project Structure

```
ARES-Cognitive-RF/
├── simulation/            # The virtual radio spectrum and jammer simulators
│   ├── ether.py           # Simulates the 32-channel spectrum and measures packet success
│   ├── jammer.py          # Jamming strategies (Sweep, Barrage, Follower, Random)
│   └── rl_environment.py  # Gym-style trial-and-error training environment
├── models/                # AI brains and training scripts
│   ├── model.py           # GRU neural network definition
│   ├── dqn_agent.py       # DQN self-learning agent definition
│   ├── train_brain.py     # Supervised training script for GRU
│   ├── train_rl.py        # Reinforcement learning training loop for DQN
│   └── test_rl.py         # Automated test suite for the RL pipeline
├── hardware_bridge/       # USB connection bridge to interface with physical ZEK keys
├── dashboard/             # Next.js web application for the Ground Control Station UI
├── requirements.txt       # Python package dependencies
└── README.md              # Project overview and user guide
```

---

## 🏗️ System Components

1. **Virtual Spectrum (`ether.py`)**: Simulates 32 radio channels. It tracks signal strength, noise levels, and detects when a transmitter and jammer occupy the same channel (a collision).
2. **Signal Jammers (`jammer.py`)**: Simulates four types of adversarial jammers:
   - **Sweep Jammer**: Scans across channels sequentially.
   - **Barrage Jammer**: Jams blocks of channels and shifts periodically.
   - **Follower Jammer**: Targets the channel the transmitter was on in the previous step.
   - **Random Jammer**: Randomly fires bursts of noise across the spectrum.
3. **Hardware Bridge (`hardware_bridge.py`)**: Manages the connection to the USB ZenTropy Key. If the hardware key is unplugged, it automatically falls back to secure software random number generation without interruption.
4. **Visual Dashboard (`dashboard/`)**: A Ground Control Station interface that displays a real-time waterfall display of the spectrum, active jammer strategies, transmission success rates, and the AI's channel predictions.

---

## 🧠 Brain Option 1: The Pattern Predictor (GRU)

This brain is trained on recorded simulation logs to recognize jammer behaviors and predict the chance of any channel being jammed in the next step.

### Performance Summary
* **Prediction Accuracy**: **98.5%** overall channel accuracy.
* **Recall Rate**: **83.8%** of actual jamming attacks successfully intercepted and avoided.
* **Inference Speed**: **0.52 ms** on CPU (well below the 5ms real-time limit, allowing rapid frequency hops).

---

## 🧠 Brain Option 2: The Self-Learning Agent (DQN)

This brain does not need any pre-recorded data. It learns dynamically by interacting with the environment, receiving rewards for successful transmissions and penalties for collisions or unnecessary hops (hop tax).

### Training & Convergence
* **Training Time**: ~21 minutes on CPU (1,000,000 steps).
* **Success Rate**: Improves from ~91% (random) to **`97.2%` Packet Delivery Ratio (PDR)** as the agent masters evasion.
* **Self-Correction**: Learns to stay on safe channels to avoid the "hop tax" unless a jammer is detected.

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Installation
* Make sure you have **Python 3.10+** and **Node.js** installed.
* Clone the repository and install the Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Running Option 1: GRU Brain
To generate training data, train the GRU predictor, and benchmark it:
```bash
# Step A: Generate a dataset of 100,000 steps
python -m simulation.generate_dataset

# Step B: Train the GRU network
python -m models.train_brain

# Step C: Evaluate accuracy and latency
python -m models.evaluate_brain
```

### 3. Running Option 2: DQN Brain
To train the self-learning agent and plot its convergence:
```bash
# Train the DQN agent (resets and runs for 2000 episodes)
python -m models.train_rl
```
This saves the trained weights to `models/dqn_brain.pth` and saves the training metrics chart to `models/dqn_training.png`.

### 4. Running the Full System (Backend Sim + Web Dashboard)
A unified script is provided to launch the simulation backend (WebSocket server) and Next.js frontend development server simultaneously:
```bash
python run.py
```
* **If using the GRU Brain (default)**: Just run `python run.py` (ensure you trained it first).
* **If using the DQN Brain**: Run `python run.py --dqn-model models/dqn_brain.pth`.
* **Hardware Options**: Use `--mock-zek` if you do not have a physical ZenTropy Key plugged in.
* **View the Dashboard**: Open your browser and navigate to `http://localhost:3000`.

### 🧪 Running Tests
To run all tests in the repository:
```bash
python -m unittest discover -s . -p "test_*.py"
```
Or run the DQN reinforcement learning unit tests specifically:
```bash
python -m unittest models.test_rl
```
