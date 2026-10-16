"""
SE(3)-Transformer Baseline for Physical Sequence Learning
简化版等变 Transformer，用于与 MCR-Former 对比
核心思想：使用等变注意力机制处理角度序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SphericalHarmonicEmbedding(nn.Module):
    """
    球谐函数嵌入 - 模拟 SE(3) 等变网络的角度编码
    将角度映射到更高维的等变特征空间
    """
    def __init__(self, l_max=3):
        super().__init__()
        self.l_max = l_max
        # 球谐函数阶数，l=0,1,2,...,l_max
        # 总维度 = sum(2l+1) for l in [0, l_max]
        self.output_dim = sum(2 * l + 1 for l in range(l_max + 1))
    
    def forward(self, angles):
        """
        angles: (batch, seq_len, 1) 角度值（弧度）
        return: (batch, seq_len, output_dim) 球谐嵌入
        """
        batch_size, seq_len, _ = angles.shape
        
        # 将角度转换为 3D 方向向量（模拟分子中的原子方向）
        # 对于 1D 角度，我们将其映射到单位圆上的点
        x = torch.cos(angles)
        y = torch.sin(angles)
        z = torch.zeros_like(angles)
        
        # 计算球谐函数特征（简化版）
        features = []
        
        # l=0 (标量)
        features.append(torch.ones_like(angles))
        
        # l=1 (向量)
        features.append(x)
        features.append(y)
        features.append(z)
        
        # l=2 (二阶张量 - 简化)
        features.append(x * x - y * y)
        features.append(2 * x * y)
        features.append(x * z)
        features.append(y * z)
        features.append(z * z - 0.5 * (x * x + y * y))
        
        # 更高阶可以继续添加...
        
        # 拼接所有特征
        embedded = torch.cat(features, dim=-1)
        
        return embedded


class EquivariantAttention(nn.Module):
    """
    等变注意力机制
    模拟 SE(3)-Transformer 的等变注意力，但适配 1D 序列
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 等变注意力需要特殊的权重初始化
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 等变约束：权重矩阵需要满足特定的对称性
        # 这里通过正交初始化近似实现
        nn.init.orthogonal_(self.q_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, d_model)
        return: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape
        
        # 投影到多头空间
        q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class EquivariantTransformerLayer(nn.Module):
    """
    等变 Transformer 层
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attention = EquivariantAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # 等变注意力 + 残差
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # 前馈网络 + 残差
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        
        return x


class SE3TransformerBaseline(nn.Module):
    """
    SE(3)-Transformer Baseline Model
    用于与 MCR-Former 对比的等变网络基线
    """
    def __init__(self, input_dim=1, d_model=64, nhead=2, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=20):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 球谐嵌入层（等变编码）
        self.spherical_embedding = SphericalHarmonicEmbedding(l_max=2)
        spherical_dim = self.spherical_embedding.output_dim
        
        # 输入投影
        self.input_projection = nn.Linear(spherical_dim, d_model)
        
        # 位置编码（可学习的，因为等变网络通常不使用固定位置编码）
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # 等变 Transformer 层
        self.transformer_layers = nn.ModuleList([
            EquivariantTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 全局池化（等变池化）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, angles, lengths):
        """
        angles: (batch, seq_len, 1) 角度值（弧度）
        lengths: (batch,) 每个序列的实际长度
        return: (batch, 1) 预测值
        """
        batch_size, seq_len, _ = angles.shape
        
        # 球谐嵌入（等变编码）
        embedded = self.spherical_embedding(angles)  # (batch, seq_len, spherical_dim)
        
        # 输入投影
        x = self.input_projection(embedded)  # (batch, seq_len, d_model)
        
        # 位置编码
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # 创建注意力掩码
        mask = torch.arange(seq_len, device=angles.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        
        # 通过 Transformer 层
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # 全局池化
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # 回归
        output = self.regression_head(x)
        
        return output.squeeze(-1)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = SE3TransformerBaseline(
        input_dim=1,
        d_model=64,
        nhead=2,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=20
    )
    
    # 测试前向传播
    batch_size = 4
    seq_len = 16
    angles = torch.randn(batch_size, seq_len, 1) * math.pi  # 随机角度
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    
    output = model(angles, lengths)
    
    print(f"输入形状：{angles.shape}")
    print(f"输出形状：{output.shape}")
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    print("✓ SE3-Transformer Baseline 测试通过")