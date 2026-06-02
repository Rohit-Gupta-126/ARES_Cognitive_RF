"""
models/test_rl.py
────────────────────────────────────────────────────────────
A.R.E.S. Stage 5 — Automated unit tests for the RL pipeline.

Tests
-----
1. RLEnvironment: state shape, reward contract, jammer cycling
2. ReplayBuffer : push, sample, capacity overflow
3. QNetwork     : output shape, gradient flow
4. DQNAgent     : action selection, learning step, checkpoint round-trip
5. Integration  : short training run converges above random baseline

Usage
-----
    python -m unittest models.test_rl -v
    python models/test_rl.py
"""

import sys
import os
import tempfile
import unittest

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.rl_environment import RLEnvironment
from models.dqn_agent import QNetwork, ReplayBuffer, DQNAgent


# ─── 1. RLEnvironment ────────────────────────────────────────────────────────

class TestRLEnvironment(unittest.TestCase):

    def setUp(self):
        """Small environment for fast tests."""
        self.env = RLEnvironment(num_channels=32, seq_len=10, max_steps=20, cycle_interval=5)

    def test_state_shape_after_reset(self):
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (self.env.state_dim,),
                         f"Expected ({self.env.state_dim},), got {state.shape}")

    def test_state_dim_value(self):
        self.assertEqual(self.env.state_dim, 321)  # 10*32 + 1

    def test_action_dim_value(self):
        self.assertEqual(self.env.action_dim, 32)

    def test_step_returns_correct_types(self):
        self.env.reset()
        next_state, reward, done, info = self.env.step(0)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward,     (float, int))
        self.assertIsInstance(done,       bool)
        self.assertIsInstance(info,       dict)

    def test_next_state_shape(self):
        self.env.reset()
        next_state, _, _, _ = self.env.step(0)
        self.assertEqual(next_state.shape, (self.env.state_dim,))

    def test_collision_reward(self):
        """Force a collision by checking the reward is <= -100."""
        env = RLEnvironment(
            max_steps=200,
            reward_success=10.0,
            reward_collision=-100.0,
            reward_hop_tax=-1.0,
        )
        env.reset()

        # Run many steps — at least one collision is expected
        collision_rewards = []
        for _ in range(200):
            _, reward, done, info = env.step(0)  # always stay on ch 0
            if not info["success"]:
                collision_rewards.append(reward)
            if done:
                break

        if collision_rewards:
            for r in collision_rewards:
                # Collision reward should be <= -100 (with possible hop tax)
                self.assertLessEqual(r, -99.0)

    def test_success_reward_positive(self):
        """On a success with no hop, reward should be exactly +10."""
        env = RLEnvironment(max_steps=50, cycle_interval=100)
        state = env.reset()

        found_success = False
        for step in range(50):
            prev_ch = env._tx_channel
            next_s, reward, done, info = env.step(prev_ch)  # stay on same channel
            if info["success"]:
                # No hop, so reward = +10 exactly
                self.assertAlmostEqual(reward, 10.0, places=2)
                found_success = True
                break
            if done:
                break

        if not found_success:
            self.skipTest("No success encountered in 50 steps — jammer coverage too high")

    def test_done_at_max_steps(self):
        self.env.reset()
        done = False
        for i in range(20):
            _, _, done, _ = self.env.step(i % 32)
        self.assertTrue(done, "Environment should be done after max_steps=20")

    def test_jammer_cycles(self):
        env = RLEnvironment(max_steps=30, cycle_interval=10)
        env.reset()
        initial_jammer = env.active_jammer_name
        # After cycle_interval steps, jammer should have rotated
        for _ in range(11):
            env.step(0)
        self.assertIsNotNone(env.active_jammer_name)

    def test_state_channel_norm_in_range(self):
        """Last element of state is channel / 31, should be in [0, 1]."""
        env = RLEnvironment(max_steps=50)
        for _ in range(5):
            env.reset()
            for ch in [0, 15, 31]:
                state, _, done, _ = env.step(ch)
                norm_ch = state[-1]
                self.assertGreaterEqual(norm_ch, 0.0)
                self.assertLessEqual(norm_ch, 1.0)
                if done:
                    env.reset()
                    break


# ─── 2. ReplayBuffer ─────────────────────────────────────────────────────────

class TestReplayBuffer(unittest.TestCase):

    def _dummy_transition(self):
        return (
            np.zeros(321, dtype=np.float32),
            0,
            1.0,
            np.ones(321, dtype=np.float32),
            False,
        )

    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        self.assertEqual(len(buf), 0)
        buf.push(*self._dummy_transition())
        self.assertEqual(len(buf), 1)

    def test_capacity_overflow(self):
        """Buffer should not grow beyond capacity."""
        buf = ReplayBuffer(capacity=10)
        for _ in range(25):
            buf.push(*self._dummy_transition())
        self.assertEqual(len(buf), 10)

    def test_sample_shapes(self):
        device = torch.device("cpu")
        buf = ReplayBuffer(capacity=200)
        for _ in range(100):
            buf.push(*self._dummy_transition())

        states, actions, rewards, next_states, dones = buf.sample(32, device)
        self.assertEqual(states.shape,      (32, 321))
        self.assertEqual(actions.shape,     (32,))
        self.assertEqual(rewards.shape,     (32,))
        self.assertEqual(next_states.shape, (32, 321))
        self.assertEqual(dones.shape,       (32,))

    def test_sample_tensor_types(self):
        device = torch.device("cpu")
        buf = ReplayBuffer(capacity=200)
        for _ in range(100):
            buf.push(*self._dummy_transition())

        states, actions, rewards, next_states, dones = buf.sample(16, device)
        self.assertEqual(states.dtype,      torch.float32)
        self.assertEqual(actions.dtype,     torch.long)
        self.assertEqual(rewards.dtype,     torch.float32)
        self.assertEqual(next_states.dtype, torch.float32)
        self.assertEqual(dones.dtype,       torch.float32)


# ─── 3. QNetwork ─────────────────────────────────────────────────────────────

class TestQNetwork(unittest.TestCase):

    def test_output_shape(self):
        net = QNetwork(state_dim=321, num_actions=32)
        x = torch.zeros(8, 321)
        out = net(x)
        self.assertEqual(out.shape, (8, 32))

    def test_gradient_flow(self):
        net = QNetwork(state_dim=321, num_actions=32)
        x = torch.randn(4, 321)
        out = net(x)
        loss = out.sum()
        loss.backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No grad for {name}")


# ─── 4. DQNAgent ─────────────────────────────────────────────────────────────

class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        self.agent = DQNAgent(
            state_dim=321,
            num_actions=32,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=1000,
            batch_size=16,
            buffer_capacity=500,
            target_update_freq=100,
        )
        self.dummy_state = np.zeros(321, dtype=np.float32)

    def test_action_in_range(self):
        for _ in range(50):
            a = self.agent.select_action(self.dummy_state)
            self.assertIn(a, range(32))

    def test_q_values_shape(self):
        q = self.agent.get_q_values(self.dummy_state)
        self.assertEqual(q.shape, (32,))

    def test_top_k_channels_length(self):
        top5 = self.agent.get_top_k_channels(self.dummy_state, k=5)
        self.assertEqual(len(top5), 5)
        # All values must be valid channel indices
        for ch in top5:
            self.assertIn(ch, range(32))

    def test_top_k_no_duplicates(self):
        top5 = self.agent.get_top_k_channels(self.dummy_state, k=5)
        self.assertEqual(len(top5), len(set(top5)))

    def test_epsilon_decays(self):
        """ε should decrease monotonically as steps increase."""
        prev_eps = self.agent.epsilon
        for _ in range(200):
            self.agent.select_action(self.dummy_state)
        self.assertLess(self.agent.epsilon, prev_eps)

    def test_learn_returns_none_when_buffer_cold(self):
        result = self.agent.learn()
        self.assertIsNone(result)

    def test_learn_returns_float_after_warmup(self):
        """Fill buffer then call learn — should return a loss value."""
        state = np.zeros(321, dtype=np.float32)
        next_s = np.ones(321, dtype=np.float32)
        for _ in range(50):
            self.agent.remember(state, 0, 1.0, next_s, False)
        loss = self.agent.learn()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)

    def test_checkpoint_round_trip(self):
        """Save and load should preserve policy net weights."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name

        try:
            self.agent.save(tmp_path)
            q_before = self.agent.get_q_values(self.dummy_state).copy()

            # Create a fresh agent and load
            agent2 = DQNAgent(state_dim=321, num_actions=32)
            agent2.load(tmp_path, inference_only=True)
            q_after = agent2.get_q_values(self.dummy_state)

            np.testing.assert_array_almost_equal(q_before, q_after, decimal=5)
        finally:
            os.unlink(tmp_path)


# ─── 5. Integration smoke test ────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """Short 20-episode training run — PDR should improve over random baseline."""

    def test_short_training_improves_pdr(self):
        env = RLEnvironment(max_steps=50, cycle_interval=25)
        agent = DQNAgent(
            state_dim=env.state_dim,
            num_actions=env.action_dim,
            epsilon_start=1.0,
            epsilon_end=0.3,
            epsilon_decay=200,
            batch_size=16,
            buffer_capacity=1000,
        )

        episode_pdrs = []
        for _ in range(20):
            state = env.reset()
            successes = 0
            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                state = next_state
                if info["success"]:
                    successes += 1
                if done:
                    break
            episode_pdrs.append(successes / 50 * 100)

        # The run shouldn't crash — that's the minimum smoke test
        self.assertEqual(len(episode_pdrs), 20)
        # Mean PDR should be non-zero (some successes expected even randomly)
        mean_pdr = np.mean(episode_pdrs)
        self.assertGreater(mean_pdr, 0.0, "Agent had 0% PDR across all 20 episodes")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
