"""
models/dqn_agent.py
────────────────────────────────────────────────────────────
A.R.E.S. Stage 5 — Deep Q-Network (DQN) Cognitive Brain.

Components
----------
QNetwork      — Compact FC network that maps state → 32 Q-values.
ReplayBuffer  — Fixed-capacity circular experience replay store.
DQNAgent      — ε-greedy agent with double-network Bellman updates,
                CUDA auto-detection, and checkpoint save/load.

Architecture (QNetwork)
-----------------------
Input(321) → Linear(512) → LayerNorm(512) → ReLU
           → Linear(256) → ReLU
           → Linear(128) → ReLU
           → Linear(32)  ← Q(s, a) for each of 32 RF channels
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─── Q-Network ───────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Compact fully-connected Q-value estimator.

    Parameters
    ----------
    state_dim   : dimensionality of the flat state vector (default 321)
    num_actions : number of discrete channel actions (default 32)
    """

    def __init__(self, state_dim: int = 321, num_actions: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[batch, state_dim]

        Returns
        -------
        Tensor[batch, num_actions]  — Q-value estimates
        """
        return self.net(x)


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-size circular experience replay buffer using pre-allocated numpy arrays.
    """

    def __init__(self, capacity: int = 50_000, state_dim: int = 321):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    # ── public API ────────────────────────────────────────────────────────
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer."""
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        """
        Draw a random mini-batch and return device-resident tensors.

        Returns
        -------
        states, actions, rewards, next_states, dones — all Tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]).to(device),
            torch.from_numpy(self.actions[indices]).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.next_states[indices]).to(device),
            torch.from_numpy(self.dones[indices]).to(device),
        )

    def __len__(self) -> int:
        return self.size


# ─── DQN Agent ───────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent with:
    • ε-greedy exploration (linear decay, 1.0 → 0.05 over epsilon_decay steps)
    • Experience replay (ReplayBuffer)
    • Separate policy + target networks (hard update every target_update_freq steps)
    • Gradient clipping (max_norm = 10.0) for training stability
    • CUDA auto-detection for RTX 2050 (or any available GPU)

    Parameters
    ----------
    state_dim          : flat state vector length (default 321)
    num_actions        : RF channels / discrete actions (default 32)
    lr                 : Adam learning rate (default 1e-3)
    gamma              : discount factor (default 0.99)
    epsilon_start      : initial exploration rate (default 1.0)
    epsilon_end        : minimum exploration rate (default 0.05)
    epsilon_decay      : steps over which ε decays linearly (default 10 000)
    batch_size         : mini-batch size for Bellman updates (default 64)
    buffer_capacity    : replay buffer size (default 50 000)
    target_update_freq : steps between hard target network syncs (default 500)
    """

    def __init__(
        self,
        state_dim: int = 321,
        num_actions: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 500,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # ε-greedy state
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # CUDA auto-detection: leverage RTX 2050 when available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network (trained) + frozen target network
        self.policy_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity, state_dim=state_dim)

    # ── Action selection ──────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """
        Choose an action via ε-greedy policy.

        During training ε decays linearly from epsilon_start → epsilon_end.
        After epsilon_decay steps the agent acts greedily.
        """
        self.steps_done += 1
        progress = min(self.steps_done / self.epsilon_decay, 1.0)
        self.epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        return int(self.get_q_values(state).argmax())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Return Q-values for all 32 channels given a state.

        Used by the HIL loop (inference) and the training loop (logging).

        Returns
        -------
        np.ndarray of shape (32,)  — raw Q-value estimates
        """
        self.policy_net.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.policy_net(t).squeeze(0).cpu().numpy()
        self.policy_net.train()
        return q

    def get_top_k_channels(self, state: np.ndarray, k: int = 5) -> list:
        """
        Return the top-k channel indices ranked by descending Q-value.

        This is the "Safe Zone" fed to the ZenTropy Key for final selection.

        Parameters
        ----------
        state : current flat state vector
        k     : size of the safe-zone candidate set (default 5)

        Returns
        -------
        list of int  — top-k channel indices, highest Q-value first
        """
        q_values = self.get_q_values(state)
        top_k = np.argsort(q_values)[::-1][:k].tolist()
        return top_k

    # ── Experience storage ────────────────────────────────────────────────
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single (s, a, r, s', done) transition."""
        self.memory.push(state, action, reward, next_state, done)

    # ── Learning step ─────────────────────────────────────────────────────
    def learn(self, learn_freq: int = 4) -> float | None:
        """
        Sample a mini-batch and execute one Bellman gradient update.

        Returns
        -------
        float — loss value, or None if the buffer is still warming up.
        """
        if self.steps_done % learn_freq != 0:
            return None
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Q(s, a) from policy network
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ · max_a' Q_target(s', a')  (masked by done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent destabilising large Q-value swings
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Hard target network sync
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ── Checkpoint helpers ────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save a full training checkpoint (policy, target, optimiser, ε)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "policy_net":   self.policy_net.state_dict(),
                "target_net":   self.target_net.state_dict(),
                "optimizer":    self.optimizer.state_dict(),
                "steps_done":   self.steps_done,
                "epsilon":      self.epsilon,
            },
            path,
        )

    def load(self, path: str, inference_only: bool = False) -> None:
        """
        Load a checkpoint.

        Parameters
        ----------
        path           : path to the .pth checkpoint file.
        inference_only : if True, only load policy weights and set eval mode.
                         Does NOT restore optimiser state or epsilon (for HIL use).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])

        if inference_only:
            self.policy_net.eval()
        else:
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.steps_done = checkpoint.get("steps_done", 0)
            self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
            self.policy_net.train()
