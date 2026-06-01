import torch
import torch.nn as nn

class JammerPredictorGRU(nn.Module):
    """
    GRU-based neural network for predicting future jamming channel states.
    Input: [Batch, SeqLen, 32] - Historical binary jammed channel vectors.
    Output: [Batch, 32] - Predicted jamming logits for the next time step.
    """
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=2, output_dim=32):
        super(JammerPredictorGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # GRU Layer
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Output linear layer to map GRU hidden state to 32 logits
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        Forward pass.
        x: tensor of shape [batch_size, seq_len, input_dim]
        Returns: logits tensor of shape [batch_size, output_dim]
        """
        # GRU returns: (output, h_n)
        # output shape: [batch_size, seq_len, hidden_dim]
        # h_n shape: [num_layers, batch_size, hidden_dim]
        gru_out, _ = self.gru(x)
        
        # Take the output of the last sequence step
        last_step_out = gru_out[:, -1, :] # [batch_size, hidden_dim]
        
        # Project to output logits
        logits = self.fc_out(last_step_out) # [batch_size, output_dim]
        return logits
