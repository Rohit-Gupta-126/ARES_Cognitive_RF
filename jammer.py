import numpy as np
import random

class BaseJammer:
    """Base class for Electronic Warfare (EW) Jammers."""
    def __init__(self, num_channels=32):
        self.num_channels = num_channels

    def reset(self):
        """Resets jammer state if necessary."""
        pass

    def get_jammed_channels(self, step_idx, tx_history=None):
        """
        Returns list of channel indices to jam at the current step.
        
        Parameters:
            step_idx (int): Current simulation step index.
            tx_history (list): List of previous tx channels selected by the transmitter.
        """
        raise NotImplementedError("Subclasses must implement get_jammed_channels")


class SweepJammer(BaseJammer):
    """
    Sweep Jammer: Jams channels sequentially.
    Includes user-requested randomness: occasionally alters step size or reverses direction.
    """
    def __init__(self, num_channels=32, initial_channel=0, step_size=1, jam_width=1, randomness_prob=0.1):
        super().__init__(num_channels)
        self.initial_channel = initial_channel
        self.base_step_size = step_size
        self.step_size = step_size
        self.jam_width = jam_width
        self.randomness_prob = randomness_prob
        self.reset()

    def reset(self):
        self.current_channel = self.initial_channel
        self.direction = 1  # 1 for forward, -1 for backward
        self.step_size = self.base_step_size

    def get_jammed_channels(self, step_idx, tx_history=None):
        # Apply user feedback: introduce slight randomness (reverse or change step size)
        if random.random() < self.randomness_prob:
            # Randomly toggle direction
            if random.random() < 0.5:
                self.direction *= -1
            # Randomly alter step size between base_step_size and base_step_size + 1
            self.step_size = random.choice([self.base_step_size, self.base_step_size + 1])
            if self.step_size == 0:
                self.step_size = 1

        # Calculate channels to jam in this step (based on current_channel and jam_width)
        jammed = []
        for offset in range(self.jam_width):
            ch = (self.current_channel + offset) % self.num_channels
            jammed.append(int(ch))

        # Advance current channel for next step
        self.current_channel = (self.current_channel + self.direction * self.step_size) % self.num_channels
        return jammed


class BarrageJammer(BaseJammer):
    """
    Barrage Jammer: Jams a wide contiguous band of channels simultaneously.
    Includes user-requested adaptability: shifts the block randomly every N steps.
    """
    def __init__(self, num_channels=32, block_size=6, shift_interval=20):
        super().__init__(num_channels)
        self.block_size = min(block_size, num_channels)
        self.shift_interval = shift_interval
        self.reset()

    def reset(self):
        self.start_channel = random.randint(0, self.num_channels - self.block_size)

    def get_jammed_channels(self, step_idx, tx_history=None):
        # Apply user feedback: shift the jammed block every N steps
        if step_idx > 0 and step_idx % self.shift_interval == 0:
            self.start_channel = random.randint(0, self.num_channels - self.block_size)

        jammed = []
        for i in range(self.block_size):
            ch = (self.start_channel + i) % self.num_channels
            jammed.append(int(ch))
        return jammed


class FollowerJammer(BaseJammer):
    """
    Follower Jammer: Detects the transmitter's previous frequency (t-1)
    and targets that channel at the current step t.
    """
    def __init__(self, num_channels=32, fallback_jammer=None):
        super().__init__(num_channels)
        # fallback_jammer is used if tx_history is empty (step 0)
        self.fallback_jammer = fallback_jammer or RandomJammer(num_channels, num_to_jam=1)

    def reset(self):
        if self.fallback_jammer:
            self.fallback_jammer.reset()

    def get_jammed_channels(self, step_idx, tx_history=None):
        if tx_history and len(tx_history) > 0:
            # Jam the transmitter's last channel (1-step delay)
            last_tx = tx_history[-1]
            if last_tx is not None and 0 <= last_tx < self.num_channels:
                return [int(last_tx)]
        
        # Fallback if no history exists (e.g. step 0)
        return self.fallback_jammer.get_jammed_channels(step_idx, tx_history)


class RandomJammer(BaseJammer):
    """Random Jammer: Jams a random selection of channels at each step."""
    def __init__(self, num_channels=32, num_to_jam=2):
        super().__init__(num_channels)
        self.num_to_jam = min(num_to_jam, num_channels)

    def get_jammed_channels(self, step_idx, tx_history=None):
        jammed = random.sample(range(self.num_channels), self.num_to_jam)
        return [int(ch) for ch in jammed]
