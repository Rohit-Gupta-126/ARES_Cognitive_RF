"""
simulation/rl_environment.py
────────────────────────────────────────────────────────────
A.R.E.S. Stage 5 — OpenAI-Gym-style RF Environment wrapper.

Wraps RFEther + all four jammer strategies into a single
step/reset interface the DQNAgent trains against.

MDP Definition
--------------
State  : (10 × 32) sliding binary jam-history  + current TX channel (normalised)
         → flattened to shape (321,)
Action : Integer 0-31 (channel to hop to)
Reward : +10 clear TX · -100 collision · -1 hop-tax (channel switch)
Done   : episode reaches max_steps
"""

import numpy as np
import random

from .ether import RFEther
from .jammer import SweepJammer, BarrageJammer, FollowerJammer, RandomJammer


class RLEnvironment:
    """
    Gym-style environment for the A.R.E.S. DQN training arena.

    Parameters
    ----------
    num_channels    : RF spectrum width (default 32)
    seq_len         : length of the sliding history window (default 10)
    max_steps       : episode horizon (default 500)
    cycle_interval  : steps between automatic jammer strategy rotations
    reward_success  : reward for a clean packet delivery
    reward_collision: penalty for a jamming collision
    reward_hop_tax  : penalty applied when the agent changes channels
    """

    # ── Public properties ──────────────────────────────────────────────────
    @property
    def state_dim(self) -> int:
        """Flat state vector length: seq_len × num_channels + 1."""
        return self.seq_len * self.num_channels + 1  # 321

    @property
    def action_dim(self) -> int:
        """Number of discrete actions (one per RF channel)."""
        return self.num_channels  # 32

    # ── Constructor ────────────────────────────────────────────────────────
    def __init__(
        self,
        num_channels: int = 32,
        seq_len: int = 10,
        max_steps: int = 500,
        cycle_interval: int = 50,
        reward_success: float = 10.0,
        reward_collision: float = -100.0,
        reward_hop_tax: float = -1.0,
    ):
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.max_steps = max_steps
        self.cycle_interval = cycle_interval

        # Reward shaping constants
        self.R_SUCCESS = reward_success
        self.R_COLLISION = reward_collision
        self.R_HOP_TAX = reward_hop_tax

        # RF physics layer
        self.ether = RFEther(num_channels=num_channels)

        # Adversarial jammer ensemble (all four strategies)
        self._jammer_pool = {
            "sweep":    SweepJammer(num_channels, step_size=1, jam_width=1, randomness_prob=0.1),
            "barrage":  BarrageJammer(num_channels, block_size=5, shift_interval=20),
            "follower": FollowerJammer(num_channels),
            "random":   RandomJammer(num_channels, num_to_jam=2),
        }
        self._jammer_names = list(self._jammer_pool.keys())

        # Runtime state (initialised by reset)
        self._history: np.ndarray = None  # type: ignore
        self._tx_history: list = []
        self._tx_channel: int = 0
        self._step_count: int = 0
        self._jammer_idx: int = 0
        self.active_jammer: object = None
        self.active_jammer_name: str = ""

    # ── Gym interface ──────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Returns
        -------
        np.ndarray of shape (state_dim,)
            The initial state observation.
        """
        self.ether.reset()
        self._step_count = 0
        self._history = np.zeros((self.seq_len, self.num_channels), dtype=np.float32)
        self._tx_channel = 0
        self._tx_history = []

        # Randomly seed the starting jammer so the agent sees diverse openers
        self._jammer_idx = random.randint(0, len(self._jammer_names) - 1)
        self._activate_jammer(self._jammer_idx)

        return self._build_state()

    def step(self, action: int):
        """
        Execute one environment step.

        Parameters
        ----------
        action : int
            Channel index (0–31) chosen by the DQN agent.

        Returns
        -------
        next_state : np.ndarray  (state_dim,)
        reward     : float
        done       : bool
        info       : dict  (jam_channels, success, active_jammer, …)
        """
        # ── Dynamic jammer cycling ─────────────────────────────────────────
        if self._step_count > 0 and self._step_count % self.cycle_interval == 0:
            self._jammer_idx = (self._jammer_idx + 1) % len(self._jammer_names)
            self._activate_jammer(self._jammer_idx)

        prev_channel = self._tx_channel
        self._tx_channel = int(action)

        # ── Causal ordering: jammer decides BEFORE seeing this hop ─────────
        jam_channels = self.active_jammer.get_jammed_channels(
            self._step_count, self._tx_history
        )

        # Now record the hop so the follower jammer can track it next step
        self._tx_history.append(self._tx_channel)
        if len(self._tx_history) > 10:
            self._tx_history.pop(0)

        # ── Physics layer ──────────────────────────────────────────────────
        result = self.ether.step(self._tx_channel, jam_channels)

        # ── Reward shaping ─────────────────────────────────────────────────
        if result["success"]:
            reward = self.R_SUCCESS
        else:
            reward = self.R_COLLISION  # -100: collision

        # Hop tax: penalise needless channel switching to learn link stability
        if self._tx_channel != prev_channel:
            reward += self.R_HOP_TAX   # -1

        # ── Slide history window ───────────────────────────────────────────
        self._history = np.roll(self._history, -1, axis=0)
        self._history[-1] = result["jammed_vector"]

        self._step_count += 1
        done = self._step_count >= self.max_steps

        next_state = self._build_state()

        info = {
            "jam_channels":   jam_channels,
            "success":        bool(result["success"]),
            "active_jammer":  self.active_jammer_name,
            "jammed_vector":  result["jammed_vector"].tolist(),
            "snr":            result["snr"],
        }

        return next_state, reward, done, info

    # ── Internal helpers ───────────────────────────────────────────────────
    def _activate_jammer(self, idx: int):
        """Switch to the jammer at position idx in the pool."""
        name = self._jammer_names[idx]
        self.active_jammer_name = name
        self.active_jammer = self._jammer_pool[name]

    def _build_state(self) -> np.ndarray:
        """
        Construct the flat state vector.

        Layout:
          [ history[0][0..31], history[1][0..31], ..., history[9][0..31],
            tx_channel_norm ]

        Returns
        -------
        np.ndarray of shape (321,), dtype float32
        """
        flat_history = self._history.flatten()                              # (320,)
        channel_norm = np.array(
            [self._tx_channel / max(self.num_channels - 1, 1)],
            dtype=np.float32
        )                                                                   # (1,)
        return np.concatenate([flat_history, channel_norm])                 # (321,)
