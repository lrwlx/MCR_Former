"""
MCR_Former_alpha.py
带有可学习对称性参数的MCR-Former模型
支持固定alpha（严格对称）和可训练alpha（自适应对称）
"""

import torch
import torch.nn as nn
import math

# ==================== 位置编码层 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
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


# ==================== MCR-Former with Learnable Alpha ====================
class MCR_Former_Alpha(nn.Module):
    """
    带有可学习对称性参数的MCR-Former模型
    
    参数:
        input_dim: 输入特征维度 (2: 2D表示, 3: 3D球坐标)
        d_model: Transformer隐藏维度
        nhead: 注意力头数量
        num_layers: Transformer编码器层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout率
        max_seq_len: 最大序列长度
        alpha: 对称性参数初始值 (默认0.5，表示严格对称)
        alpha_trainable: alpha是否可训练 (默认False，保持固定值)
        alpha_fixed: 指定固定alpha值 (当alpha_trainable=False时有效)
    """
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_seq_len=30,
                 alpha=0.5, alpha_trainable=False, alpha_fixed=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.alpha_trainable = alpha_trainable
        
        # 设置alpha参数
        if alpha_fixed is not None:
            # 使用指定的固定alpha值
            self.alpha_value = alpha_fixed
            self.alpha_trainable = False
        else:
            self.alpha_value = alpha
        
        if alpha_trainable:
            # 可训练的alpha参数
            # 使用logit确保alpha在[0,1]范围内，初始值为指定alpha
            self.logit_alpha = nn.Parameter(torch.logit(torch.tensor(self.alpha_value)))
        else:
            # 固定的alpha参数
            self.register_buffer('alpha', torch.tensor(self.alpha_value))
        
        # 1. 最小完备表示嵌入层
        self.mcr_embedding = nn.Linear(input_dim, d_model)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 3. 共享Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # 提高训练稳定性
        )
        self.shared_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
    def _get_alpha(self):
        """获取当前的alpha值"""
        if self.alpha_trainable:
            # 可训练alpha，通过sigmoid确保在[0,1]范围内
            return torch.sigmoid(self.logit_alpha)
        else:
            # 固定alpha
            return self.alpha
    
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
        前向传播，支持可学习的对称性参数
        
        参数:
            x: 输入序列 [batch_size, seq_len, input_dim]
            lengths: 序列实际长度 [batch_size]
            
        返回:
            预测值 [batch_size]
        """
        mask = self._create_mask(x, lengths)
        
        # 路径A：编码原始序列
        feat_forward = self._encode_sequence(x, mask)
        
        # 路径B：编码反转序列
        x_reversed = torch.flip(x, dims=[1])
        feat_backward = self._encode_sequence(x_reversed, mask)
        
        # 获取当前alpha值
        alpha = self._get_alpha()
        
        # 对称池化：加权融合
        # 当alpha=0.5时严格对称，alpha=1.0时完全偏向原始序列
        fused_feature = alpha * feat_forward + (1 - alpha) * feat_backward
        
        # 回归预测
        return self.regression_head(fused_feature).squeeze(-1)
    
    def get_alpha_value(self):
        """获取当前alpha值（标量）"""
        alpha = self._get_alpha()
        if isinstance(alpha, torch.Tensor):
            return alpha.item()
        return float(alpha)


# ==================== 预定义模型变体 ====================

class MCR_Former_Alpha_2D(MCR_Former_Alpha):
    """用于复合材料的2D版本，带可学习alpha"""
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, max_seq_len=30, alpha=0.5, alpha_trainable=False):
        super().__init__(
            input_dim=2,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            alpha=alpha,
            alpha_trainable=alpha_trainable
        )


class MCR_Former_Alpha_3D(MCR_Former_Alpha):
    """用于量子自旋的3D版本，带可学习alpha"""
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, max_seq_len=30, alpha=0.5, alpha_trainable=False):
        super().__init__(
            input_dim=3,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            alpha=alpha,
            alpha_trainable=alpha_trainable
        )


class MCR_Former_Strict_Symmetric(MCR_Former_Alpha):
    """严格对称版本（固定alpha=0.5）"""
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_seq_len=30):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            alpha=0.5,
            alpha_trainable=False,
            alpha_fixed=0.5  # 明确指定固定值
        )


class MCR_Former_Forward_Only(MCR_Former_Alpha):
    """只使用正向序列（固定alpha=1.0）"""
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_seq_len=30):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            alpha=1.0,
            alpha_trainable=False,
            alpha_fixed=1.0
        )


class MCR_Former_Reverse_Only(MCR_Former_Alpha):
    """只使用反向序列（固定alpha=0.0）"""
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_seq_len=30):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            alpha=0.0,
            alpha_trainable=False,
            alpha_fixed=0.0
        )


# ==================== 工具函数 ====================

def create_mcr_former_alpha(config):
    """
    根据配置创建MCR-Former Alpha模型
    
    参数:
        config: 配置字典，包含模型参数
        
    返回:
        MCR_Former_Alpha模型实例
    """
    # 提取alpha相关参数
    alpha = config.get('alpha', 0.5)
    alpha_trainable = config.get('alpha_trainable', False)
    alpha_fixed = config.get('alpha_fixed', None)
    
    # 创建模型
    model = MCR_Former_Alpha(
        input_dim=config.get('input_dim', 2),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        alpha=alpha,
        alpha_trainable=alpha_trainable,
        alpha_fixed=alpha_fixed
    )
    
    return model


def analyze_alpha_dynamics(model, dataloader, device):
    """
    分析模型在数据上的alpha值动态
    
    参数:
        model: MCR_Former_Alpha模型
        dataloader: 数据加载器
        device: 设备
        
    返回:
        alpha_values: alpha值列表
        predictions: 预测值
    """
    model.eval()
    alpha_values = []
    predictions = []
    
    with torch.no_grad():
        for sequences, lengths_batch, targets, _ in dataloader:
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            
            # 获取当前alpha值
            alpha = model._get_alpha()
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            alpha_values.append(alpha)
            
            # 进行预测
            preds = model(sequences, lengths_batch)
            predictions.extend(preds.cpu().numpy())
    
    return alpha_values, predictions

'''
# 示例1：创建严格对称模型（alpha固定为0.5，不可训练）
from MCR_Former_alpha import MCR_Former_Alpha

config_v13 = {
    'input_dim': 2,
    'd_model': 64,
    'nhead': 2,
    'num_layers': 3,
    'dim_feedforward': 128,
    'dropout': 0.08,
    'max_seq_len': 20,
    'alpha': 0.5,  # 初始值
    'alpha_trainable': False  # 不可训练
}

model_v13 = MCR_Former_Alpha(**config_v13)

# 示例2：创建可学习对称性的量子自旋模型
config_quantum = {
    'input_dim': 3,
    'd_model': 64,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'max_seq_len': 22,
    'alpha': 0.5,  # 初始值
    'alpha_trainable': True  # 可训练
}

model_quantum = MCR_Former_Alpha(**config_quantum)

# 示例3：使用预定义变体
from MCR_Former_alpha import (
    MCR_Former_Alpha_2D,
    MCR_Former_Strict_Symmetric,
    MCR_Former_Forward_Only
)

# 2D版本，可学习alpha
model_2d_learnable = MCR_Former_Alpha_2D(
    d_model=64,
    nhead=2,
    alpha_trainable=True
)

# 严格对称版本（alpha固定为0.5）
model_strict = MCR_Former_Strict_Symmetric(
    input_dim=2,
    d_model=64,
    nhead=2
)

# 只使用正向序列
model_forward_only = MCR_Former_Forward_Only(
    input_dim=2,
    d_model=64,
    nhead=2
)

# 示例4：在训练过程中监控alpha值
def train_with_alpha_monitoring(model, train_loader, num_epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    alpha_history = []
    
    for epoch in range(num_epochs):
        for batch_idx, (sequences, lengths_batch, targets, _) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences, lengths_batch)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            # 记录alpha值（如果是可训练的）
            if model.alpha_trainable:
                current_alpha = model.get_alpha_value()
                alpha_history.append(current_alpha)
        
        print(f"Epoch {epoch+1}, Alpha: {model.get_alpha_value():.4f}")
    
    return alpha_history

# 示例5：分析alpha值在测试集上的分布
def test_alpha_distribution(model, test_loader):
    model.eval()
    alpha_values = []
    
    with torch.no_grad():
        for sequences, lengths_batch, _, _ in test_loader:
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            
            # 获取alpha值
            alpha = model._get_alpha()
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            alpha_values.append(alpha)
    
    print(f"Alpha统计: 均值={np.mean(alpha_values):.4f}, "
          f"标准差={np.std(alpha_values):.4f}, "
          f"范围=[{min(alpha_values):.4f}, {max(alpha_values):.4f}]")



'''


