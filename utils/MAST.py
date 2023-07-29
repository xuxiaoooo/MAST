import torch
import torch.nn as nn

class ALL_MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim1=1024, hidden_dim2=512):
        super(ALL_MLP, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU()
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(hidden_dim1, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, dropout_rate=0.2):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=3, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.bilstm(x)
        x = x.squeeze(1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class Channel(nn.Module):
    def __init__(self, input_dim=2000, hidden_dim1=2048, hidden_dim2=1024, N=3, dropout_rate=0.5):
        super(Channel, self).__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim1)
        self.all_mlp_modules = nn.ModuleList([ALL_MLP(input_dim=hidden_dim1, hidden_dim1=hidden_dim2) for _ in range(N)])
        self.bilstm_module = BiLSTM(input_dim=hidden_dim1, hidden_dim=hidden_dim2, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.fc_input(x)
        for mlp in self.all_mlp_modules:
            x = mlp(x)
        x = self.bilstm_module(x)
        return x
    
class TransformerModule(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim, dropout_rate=0.1):
        super(TransformerModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output

class MAST(nn.Module):
    def __init__(self, input_dim=2000, hidden_dim1=2048, hidden_dim2=2048, hidden_dim3=2048, dropout_rate=0.15):
        super(MAST, self).__init__()
        ff_hidden_dim1 = 1024
        self.channel = Channel(input_dim=input_dim, hidden_dim1=hidden_dim1)
        self.multihead_attention1 = TransformerModule(num_heads=16, input_dim=hidden_dim1, hidden_dim=ff_hidden_dim1, dropout_rate=dropout_rate)
        self.multihead_attention2 = TransformerModule(num_heads=8, input_dim=hidden_dim2, hidden_dim=hidden_dim3, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        outputs1 = []
        for i in range(24):
            temp = [self.channel(x[:, i, j, :]) for j in range(16)]
            temp = torch.stack(temp, dim=1)
            temp = temp.transpose(0, 1)
            outputs1.append(self.multihead_attention1(temp)[-1])

        outputs1 = torch.stack(outputs1, dim=1)
        outputs1 = outputs1.transpose(0, 1)
        outputs2 = self.multihead_attention2(outputs1)
        outputs2 = self.dropout(outputs2)
        outputs2 = torch.mean(outputs2, dim=0)
        outputs2 = self.fc1(outputs2).squeeze(-1)

        return outputs2
