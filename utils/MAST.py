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
    
class MultiHeadAttentionModule(nn.Module):
    def __init__(self, num_heads, input_dim, output_dim):
        super(MultiHeadAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output[-1]
        output = self.fc(attn_output)
        return output

class MAST(nn.Module):
    def __init__(self, input_dim=2000, hidden_dim1=2048, hidden_dim2=1008, hidden_dim3=512, dropout_rate=0.5):
        super(MAST, self).__init__()
        assert hidden_dim1 % 16 == 0, "hidden_dim1 must be divisible by 16"
        assert hidden_dim2 % 24 == 0, "hidden_dim2 must be divisible by 24"
        self.channel = Channel(input_dim=input_dim, hidden_dim1=hidden_dim1)
        self.multihead_attention1 = MultiHeadAttentionModule(num_heads=16, input_dim=hidden_dim1, output_dim=hidden_dim2)
        self.multihead_attention2 = MultiHeadAttentionModule(num_heads=24, input_dim=hidden_dim2, output_dim=hidden_dim3)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        batch_size = x.size(0)
        print(x.size())
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
        outputs2 = self.fc1(outputs2)

        return outputs2
