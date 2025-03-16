#%%
import torch
import torch.nn as nn


#%%
import torch
import torch.nn as nn

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1):
        super(AttentionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.reshape(-1,x.shape[-1])
        h = self.fc1(x)
        h = h.unsqueeze(0) 
        attn_out, _ = self.attention(h, h, h)
        h = self.norm(h + attn_out)
        h = h.squeeze(0)
        out = self.fc2(h)
        out = out.squeeze(0)
        return out

#%%
if __name__ == "__main__":
    input_tensor = torch.randn(69)  
    model = AttentionMLP(input_dim=69, hidden_dim=128, output_dim=10, num_heads=4)
    output = model(input_tensor)
    print("Output shape:", output.shape)

# %%
