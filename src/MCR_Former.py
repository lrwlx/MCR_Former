"""
MCR-Former: Minimal Complete Representations Enable Cross-Domain Physical Sequence Learning
统一模型架构，支持任意维度的最小完备表示
"""
import torch
import torch.nn as nn
import math

# ==================== 位置编码层 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x, lengths=None):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# ==================== 统一MCR-Former模型 ====================
class MCR_Former(nn.Module):
    """
    统一MCR-Former模型，支持任意维度的最小完备表示
    
    参数:
        input_dim: 输入特征的维度
                   2: 2D最小完备表示 (sin2θ, cos2θ) - 用于复合材料
                   3: 3D球坐标表示 (x, y, z) - 用于量子自旋系统
                   n: 其他维度的最小完备表示
        d_model: Transformer隐藏维度
        nhead: 注意力头数量
        num_layers: Transformer编码器层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout率
        max_seq_len: 最大序列长度
    """
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_seq_len=20):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        
        # 1. 最小完备表示嵌入层 - 维度由input_dim决定
        self.mcr_embedding = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 3. 共享Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.shared_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
    def _create_mask(self, x, lengths):
        """创建序列掩码"""
        if lengths is None:
            return None
        batch_size, seq_len = x.size(0), x.size(1)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        for i, length in enumerate(lengths):
            actual_length = min(length.item(), seq_len)
            if actual_length < seq_len:
                mask[i, actual_length:] = True
        return mask
    
    def _encode_sequence(self, x, mask=None):
        """
        序列编码核心函数
        """
        # MCR嵌入与位置编码
        x = self.mcr_embedding(x)
        x = self.pos_encoder(x)
        
        # 通过共享Transformer编码器
        if mask is not None:
            x = self.shared_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.shared_encoder(x)
        
        # 序列聚合（加权平均）
        if mask is not None:
            valid_mask = (~mask).float().unsqueeze(-1)
            x_pooled = torch.sum(x * valid_mask, dim=1) / (torch.sum(valid_mask, dim=1) + 1e-8)
        else:
            x_pooled = torch.mean(x, dim=1)
            
        return x_pooled
    
    def forward(self, x, lengths=None):
        """
        严格对称前向传播
        1. 同一套权重分别编码原始序列和反转序列
        2. 对称池化（平均）保证严格逆向不变性
        """
        mask = self._create_mask(x, lengths)
        
        # 路径A：编码原始序列
        feat_forward = self._encode_sequence(x, mask)
        
        # 路径B：编码反转序列
        x_reversed = torch.flip(x, dims=[1])
        feat_backward = self._encode_sequence(x_reversed, mask)
        
        # Reversal-Invariant Pooling
        fused_feature = (feat_forward + feat_backward) / 2.0
        
        # 回归预测
        return self.regression_head(fused_feature).squeeze(-1)


# ==================== 为向后兼容保留的别名 ====================
# 这些别名允许现有代码继续使用，而无需修改
class MCR_Former_2D(MCR_Former):
    """用于复合材料的角度预测，使用2D最小完备表示"""
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, max_seq_len=20):
        super().__init__(input_dim=2, d_model=d_model, nhead=nhead, 
                        num_layers=num_layers, dim_feedforward=dim_feedforward,
                        dropout=dropout, max_seq_len=max_seq_len)

class MCR_Former_3D(MCR_Former):
    """用于量子自旋系统的预测，使用3D球坐标表示"""
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, max_seq_len=22):
        super().__init__(input_dim=3, d_model=d_model, nhead=nhead,
                        num_layers=num_layers, dim_feedforward=dim_feedforward,
                        dropout=dropout, max_seq_len=max_seq_len)
