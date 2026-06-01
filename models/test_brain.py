import unittest
import torch
import numpy as np
import os
from models.model import JammerPredictorGRU
from models.train_brain import RFDataset

class TestBrainModel(unittest.TestCase):
    def test_model_shapes(self):
        batch_size = 16
        seq_len = 10
        input_dim = 32
        hidden_dim = 64
        num_layers = 2
        
        model = JammerPredictorGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=input_dim
        )
        
        # Test input shape [Batch, SeqLen, InputDim]
        dummy_input = torch.randn(batch_size, seq_len, input_dim)
        output = model(dummy_input)
        
        # Output should be [Batch, InputDim]
        self.assertEqual(output.shape, (batch_size, input_dim))

    def test_dataset_wrapper(self):
        X = np.random.binomial(1, 0.1, (100, 10, 32)).astype(np.float32)
        Y = np.random.binomial(1, 0.1, (100, 32)).astype(np.float32)
        
        dataset = RFDataset(X, Y)
        self.assertEqual(len(dataset), 100)
        
        x_sample, y_sample = dataset[0]
        self.assertEqual(x_sample.shape, (10, 32))
        self.assertEqual(y_sample.shape, (32,))
        self.assertEqual(x_sample.dtype, torch.float32)
        self.assertEqual(y_sample.dtype, torch.float32)

    def test_onnx_export_flow(self):
        # Test that we can export a model to ONNX without syntax errors
        onnx_test_path = "models/test_brain.onnx"
        if os.path.exists(onnx_test_path):
            os.remove(onnx_test_path)
            
        try:
            model = JammerPredictorGRU(input_dim=32, hidden_dim=32, num_layers=1, output_dim=32)
            model.eval()
            
            dummy_input = torch.zeros(1, 10, 32)
            torch.onnx.export(
                model,
                dummy_input,
                onnx_test_path,
                export_params=True,
                opset_version=18,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            self.assertTrue(os.path.exists(onnx_test_path))
            self.assertTrue(os.path.getsize(onnx_test_path) > 0)
        finally:
            if os.path.exists(onnx_test_path):
                os.remove(onnx_test_path)

if __name__ == '__main__':
    unittest.main()
