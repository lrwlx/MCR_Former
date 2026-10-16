"""
MCR_Former_concat.py
双路径特征拼接融合的MCR-Former模型
通过拼接前向和反向路径特征实现更复杂的特征融合
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


# ==================== MCR-Former with Feature Concatenation ====================
class MCR_Former_Concat(nn.Module):
    """
    双路径特征拼接融合的MCR-Former模型
    
    参数:
        input_dim: 输入特征维度 (2: 2D表示, 3: 3D球坐标)
        d_model: Transformer隐藏维度
        nhead: 注意力头数量
        num_layers: Transformer编码器层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout率
        max_seq_len: 最大序列长度
        norm_first: 是否使用LayerNorm前置 (默认True，提高稳定性)
    """
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_seq_len=30,
                 norm_first=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        
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
            norm_first=norm_first  # 提高训练稳定性
        )
        self.shared_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. 特征融合层 - 拼接前向和后向特征后进行非线性变换
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU()
        )
        
        # 5. 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(d_model // 2, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
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
        双路径特征拼接前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, input_dim]
            lengths: 序列实际长度 [batch_size]
            
        返回:
            预测值 [batch_size]
        """
        mask = self._create_mask(x, lengths)
        
        # 路径A：编码原始序列（前向）
        feat_forward = self._encode_sequence(x, mask)
        
        # 路径B：编码反转序列（后向）
        x_reversed = torch.flip(x, dims=[1])
        feat_backward = self._encode_sequence(x_reversed, mask)
        
        # 特征拼接与融合
        # 将前向和后向特征拼接，然后通过非线性变换
        fused_feature = torch.cat([feat_forward, feat_backward], dim=1)
        fused_feature = self.fusion_layer(fused_feature)
        
        # 回归预测
        return self.regression_head(fused_feature).squeeze(-1)


# ==================== 预定义模型变体 ====================

class MCR_Former_Concat_2D(MCR_Former_Concat):
    """用于复合材料的2D版本，特征拼接融合"""
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, max_seq_len=30, norm_first=True):
        super().__init__(
            input_dim=2,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            norm_first=norm_first
        )


class MCR_Former_Concat_3D(MCR_Former_Concat):
    """用于量子自旋的3D版本，特征拼接融合"""
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, max_seq_len=30, norm_first=True):
        super().__init__(
            input_dim=3,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            norm_first=norm_first
        )


class MCR_Former_Concat_Large(MCR_Former_Concat):
    """大型特征拼接融合模型"""
    def __init__(self, input_dim=2, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=30, norm_first=True):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            norm_first=norm_first
        )


# ==================== 工具函数 ====================

def create_mcr_former_concat(config):
    """
    根据配置创建MCR-Former Concat模型
    
    参数:
        config: 配置字典，包含模型参数
        
    返回:
        MCR_Former_Concat模型实例
    """
    model = MCR_Former_Concat(
        input_dim=config.get('input_dim', 2),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        norm_first=config.get('norm_first', True)
    )
    
    return model


def analyze_feature_similarity(model, dataloader, device):
    """
    分析前向和后向特征的相似性
    
    参数:
        model: MCR_Former_Concat模型
        dataloader: 数据加载器
        device: 设备
        
    返回:
        cosine_similarities: 余弦相似度列表
        euclidean_distances: 欧几里得距离列表
    """
    model.eval()
    cosine_similarities = []
    euclidean_distances = []
    
    with torch.no_grad():
        for sequences, lengths_batch, _, _ in dataloader:
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            
            # 获取前向和后向特征
            mask = model._create_mask(sequences, lengths_batch)
            feat_forward = model._encode_sequence(sequences, mask)
            feat_backward = model._encode_sequence(torch.flip(sequences, dims=[1]), mask)
            
            # 计算余弦相似度
            for f, b in zip(feat_forward, feat_backward):
                cos_sim = torch.nn.functional.cosine_similarity(f.unsqueeze(0), b.unsqueeze(0))
                cosine_similarities.append(cos_sim.item())
                
                # 计算欧几里得距离
                euclidean_dist = torch.norm(f - b, p=2)
                euclidean_distances.append(euclidean_dist.item())
    
    return cosine_similarities, euclidean_distances


# ==================== 模型比较工具 ====================

class MCR_Former_ModelFamily:
    """MCR-Former模型家族管理类"""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        创建指定类型的MCR-Former模型
        
        参数:
            model_type: 模型类型，可选值:
                'concat': 特征拼接融合
                'alpha': 可学习alpha融合
                'strict': 严格对称融合 (alpha=0.5)
                'forward_only': 只使用前向序列
                'vanilla': 原版MCR-Former
            **kwargs: 模型参数
            
        返回:
            模型实例
        """
        if model_type == 'concat':
            return MCR_Former_Concat(**kwargs)
        elif model_type == 'alpha':
            from MCR_Former_alpha import MCR_Former_Alpha
            return MCR_Former_Alpha(**kwargs)
        elif model_type == 'strict':
            from MCR_Former_alpha import MCR_Former_Strict_Symmetric
            return MCR_Former_Strict_Symmetric(**kwargs)
        elif model_type == 'forward_only':
            from MCR_Former_alpha import MCR_Former_Forward_Only
            return MCR_Former_Forward_Only(**kwargs)
        elif model_type == 'vanilla':
            from MCR_Former import MCR_Former
            return MCR_Former(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def compare_models(models, test_loader, device, criterion=nn.MSELoss()):
        """
        比较多个模型的性能
        
        参数:
            models: 模型字典 {name: model}
            test_loader: 测试数据加载器
            device: 设备
            criterion: 损失函数
            
        返回:
            results: 性能结果字典
        """
        results = {}
        
        for name, model in models.items():
            model.eval()
            total_loss = 0.0
            predictions = []
            targets = []
            
            with torch.no_grad():
                for sequences, lengths_batch, target_values, _ in test_loader:
                    sequences = sequences.to(device)
                    lengths_batch = lengths_batch.to(device)
                    target_values = target_values.to(device)
                    
                    outputs = model(sequences, lengths_batch)
                    loss = criterion(outputs, target_values)
                    total_loss += loss.item() * sequences.size(0)
                    
                    predictions.extend(outputs.cpu().numpy())
                    targets.extend(target_values.cpu().numpy())
            
            avg_loss = total_loss / len(test_loader.dataset)
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            r2 = r2_score(targets, predictions)
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            
            results[name] = {
                'loss': avg_loss,
                'r2': r2,
                'rmse': rmse,
                'model_type': model.__class__.__name__
            }
            
            print(f"{name}: Loss={avg_loss:.6f}, R²={r2:.4f}, RMSE={rmse:.6f}")
        
        return results


# 导入评估指标
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np