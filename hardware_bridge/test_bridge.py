import unittest
import numpy as np
import os
from hardware_bridge.hardware_bridge import ZenTropyBridge

class TestZenTropyBridge(unittest.TestCase):
    def test_mock_mode_entropy(self):
        # In mock mode, the bridge should not attempt to connect to serial ports
        bridge = ZenTropyBridge(mock=True)
        self.assertFalse(bridge.is_connected)
        self.assertEqual(bridge.entropy_source, "SOFTWARE_ENTROPY")
        
        # Test raw entropy generation
        raw = bridge.read_raw_entropy(num_bytes=32)
        self.assertEqual(len(raw), 32)
        self.assertIsInstance(raw, bytes)
        
        # Test channel selection
        safe_channels = [3, 7, 12, 25]
        for _ in range(50):
            ch = bridge.get_random_channel(safe_channels)
            self.assertIn(ch, safe_channels)
            
        bridge.close()

    def test_fallback_on_invalid_port(self):
        # Trying to connect to a nonexistent port should fail gracefully
        # and enter software fallback mode
        bridge = ZenTropyBridge(port="COM999", mock=False)
        self.assertFalse(bridge.is_connected)
        self.assertEqual(bridge.entropy_source, "SOFTWARE_ENTROPY")
        
        raw = bridge.read_raw_entropy(num_bytes=16)
        self.assertEqual(len(raw), 16)
        
        safe_channels = [1, 2, 3]
        ch = bridge.get_random_channel(safe_channels)
        self.assertIn(ch, safe_channels)
        
        bridge.close()

    def test_shake_uniformity(self):
        # Generates a large sample of selected channels and checks that
        # the indices are uniformly distributed
        bridge = ZenTropyBridge(mock=True)
        safe_channels = list(range(10)) # 10 channels
        
        samples = []
        for _ in range(10000):
            samples.append(bridge.get_random_channel(safe_channels))
            
        # Count frequencies
        counts = np.bincount(samples, minlength=10)
        expected_freq = 10000 / 10
        
        # Check that each channel is within 15% of the expected frequency (1000)
        # This is a soft chi-squared verification
        for ch, count in enumerate(counts):
            percentage_diff = abs(count - expected_freq) / expected_freq
            self.assertTrue(
                percentage_diff < 0.15,
                f"Channel {ch} has frequency {count}, which deviates from uniform distribution by {percentage_diff*100:.2f}%"
            )
            
        bridge.close()

if __name__ == '__main__':
    unittest.main()
