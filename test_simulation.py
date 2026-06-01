import unittest
import numpy as np
import os
from ether import RFEther
from jammer import SweepJammer, BarrageJammer, FollowerJammer, RandomJammer
from generate_dataset import generate_simulation_data

class TestRFEther(unittest.TestCase):
    def test_initialization(self):
        ether = RFEther(num_channels=32, signal_power=20.0, jammer_power=30.0, noise_floor=1.0)
        self.assertEqual(ether.num_channels, 32)
        self.assertEqual(ether.signal_power, 20.0)
        self.assertEqual(ether.jammer_power, 30.0)
        self.assertEqual(ether.noise_floor, 1.0)
        
    def test_snr_calculation_clean_channel(self):
        # A clean channel has only background noise (around 1.0)
        # SNR = signal_power / (noise_floor + fluctuations)
        # With threshold_db = 5.0, linear threshold is 3.16. 
        # signal_power = 20.0, noise_floor = 1.0. SNR should be around 20.0 / 1.0 = 20.0. 
        # This is > 3.16, so success should be True.
        ether = RFEther(num_channels=32, signal_power=20.0, jammer_power=30.0, noise_floor=1.0, snr_threshold_db=5.0)
        res = ether.step(tx_channel=5, jam_channels=[])
        
        self.assertIsNotNone(res['snr'])
        self.assertTrue(res['success'])
        # Noise should be roughly 1.0 (with small fluctuations)
        self.assertAlmostEqual(res['channel_states'][5], 1.0, delta=0.25)
        self.assertEqual(res['jammed_vector'][5], 0.0)

    def test_snr_calculation_jammed_channel(self):
        # Jammed channel has noise = noise_floor + jammer_power = 1.0 + 30.0 = 31.0
        # SNR = 20.0 / 31.0 = 0.645
        # This is < 3.16, so success should be False.
        ether = RFEther(num_channels=32, signal_power=20.0, jammer_power=30.0, noise_floor=1.0, snr_threshold_db=5.0)
        res = ether.step(tx_channel=5, jam_channels=[5])
        
        self.assertFalse(res['success'])
        self.assertEqual(res['jammed_vector'][5], 1.0)
        self.assertTrue(res['snr'] < 1.0)


class TestJammers(unittest.TestCase):
    def test_sweep_jammer(self):
        # Set randomness_prob to 0 to test deterministic sweeping first
        jammer = SweepJammer(num_channels=32, initial_channel=0, step_size=1, jam_width=1, randomness_prob=0.0)
        
        # Should start at channel 0
        self.assertEqual(jammer.get_jammed_channels(0), [0])
        # Next should be 1
        self.assertEqual(jammer.get_jammed_channels(1), [1])
        # Next should be 2
        self.assertEqual(jammer.get_jammed_channels(2), [2])
        
        # Test jam_width > 1
        jammer_wide = SweepJammer(num_channels=32, initial_channel=0, step_size=2, jam_width=3, randomness_prob=0.0)
        # Should jam [0, 1, 2]
        self.assertEqual(jammer_wide.get_jammed_channels(0), [0, 1, 2])
        # Step size is 2, next channel starts at 2, jams [2, 3, 4]
        self.assertEqual(jammer_wide.get_jammed_channels(1), [2, 3, 4])

    def test_barrage_jammer(self):
        # Test that block size is respected
        jammer = BarrageJammer(num_channels=32, block_size=5, shift_interval=10)
        
        # Get first block
        jammed_0 = jammer.get_jammed_channels(0)
        self.assertEqual(len(jammed_0), 5)
        # Ensure block is contiguous
        diffs = np.diff(jammed_0)
        self.assertTrue(np.all(diffs == 1) or (31 in jammed_0 and 0 in jammed_0)) # contiguous or wrap-around
        
        # Should stay the same until step 10
        jammed_5 = jammer.get_jammed_channels(5)
        self.assertEqual(jammed_0, jammed_5)

    def test_follower_jammer(self):
        jammer = FollowerJammer(num_channels=32)
        
        # On first step (no history), should fall back to random
        jammed_0 = jammer.get_jammed_channels(0, tx_history=[])
        self.assertEqual(len(jammed_0), 1)
        
        # With history, should follow the last tx channel
        jammed_1 = jammer.get_jammed_channels(1, tx_history=[15])
        self.assertEqual(jammed_1, [15])
        
        jammed_2 = jammer.get_jammed_channels(2, tx_history=[15, 23])
        self.assertEqual(jammed_2, [23])


class TestDatasetGeneration(unittest.TestCase):
    def test_generation_dimensions(self):
        output_file = "test_rf_dataset.npz"
        if os.path.exists(output_file):
            os.remove(output_file)
            
        try:
            seq_len = 8
            num_steps = 1000
            X, Y = generate_simulation_data(num_steps=num_steps, seq_len=seq_len, output_path=output_file)
            
            self.assertTrue(os.path.exists(output_file))
            self.assertEqual(X.shape, (num_steps - seq_len, seq_len, 32))
            self.assertEqual(Y.shape, (num_steps - seq_len, 32))
            
            # Load and verify contents
            with np.load(output_file) as data:
                self.assertTrue("X" in data)
                self.assertTrue("Y" in data)
                self.assertEqual(data["X"].shape, X.shape)
                self.assertEqual(data["Y"].shape, Y.shape)
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
