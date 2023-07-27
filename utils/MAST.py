import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(2, 1)

    def forward(self, x):
        batch_size = x.size(0)

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.depth)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(output)
        return output

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 24)  # Change output dimension to 24
        )

    def forward(self, x):
        batch_size, num_channels, feature_dim = x.shape
        x = x.reshape(-1, feature_dim)  # Change view to reshape
        x = self.mlp(x)
        x = x.view(batch_size, num_channels, -1)  # Retain the 3 dimensions
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(24, 24, 3, padding=1),  # Change input and output channels to 24
            nn.ReLU(),
            nn.Conv1d(24, 24, 3, padding=1),  # Change input and output channels to 24
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size, num_channels, feature_dim = x.shape
        x = x.view(batch_size * num_channels, feature_dim, -1)  # Flatten the first 2 dimensions
        x = self.cnn(x)
        x = x.view(batch_size, num_channels, -1)
        return x

class LSTMUnit(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMUnit, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 24)  # Add a FC layer to change output dimension to 24

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.fc(hn[-2,:,:] + hn[-1,:,:])  # Apply FC on the summation
        return x.unsqueeze(2)  # Add an extra dimension for attention

class MAST(nn.Module):
    def __init__(self):
        super(MAST, self).__init__()
        self.mlp = MLP()
        self.cnn = CNN()
        self.lstm_unit = LSTMUnit(24, 16, 1)  # Change input size to 24
        self.transformer_unit16 = MultiheadAttention(24, 16)  # Change d_model to 24
        self.transformer_unit24 = MultiheadAttention(24, 24)  # Keep d_model as 24
        self.final_fc = nn.Sequential(
            nn.Linear(24 * 16, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        batch_size, seq_len, num_channels, feature_dim = x.shape
        transformer_outputs16 = []
        transformer_outputs24 = []

        for i in range(seq_len):
            mlp_outputs = self.mlp(x[:, i, :, :])  # Apply MLP on each time step
            mlp_outputs = mlp_outputs.view(batch_size, num_channels, -1)
            cnn_outputs = self.cnn(mlp_outputs)
            lstm_outputs = self.lstm_unit(cnn_outputs)
            lstm_outputs = lstm_outputs.view(batch_size, num_channels, -1).permute(0, 2, 1).contiguous()

            channel_outputs = []
            for j in range(num_channels):
                outputs16 = self.transformer_unit16(lstm_outputs[:,:,j].unsqueeze(2))
                channel_outputs.append(outputs16)
            channel_outputs = torch.cat(channel_outputs, dim=2)  # Concatenate results of all channels
            transformer_outputs16.append(channel_outputs)

        transformer_outputs16 = torch.stack(transformer_outputs16, dim=1)  # Stack results of all time steps
        transformer_outputs24 = self.transformer_unit24(transformer_outputs16.view(batch_size, seq_len, -1))

        final_outputs = self.final_fc(transformer_outputs24.view(batch_size, -1))

        return final_outputs
