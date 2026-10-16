# Baseline for Table 1: self-contained re-implementation of Informer, trained in this work
# on the same 3,000-sequence 16-ply dataset and the same 80/20 split (test_size=0.2,
# random_state=42) as MCR-Former. Original working copy: 07_消融实验/Informer.py.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import time
import os
import math

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 修正的ProbSparse注意力机制 - 简化版
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.attn_dropout = nn.Dropout(attention_dropout)
        
    def _prob_qk(self, Q, K, sample_k, n_top):
        # Q: [B, H, L, D], K: [B, H, S, D]
        B, H, L, E = Q.shape
        _, _, S, _ = K.shape
        
        # 随机采样K
        K_expand = K.unsqueeze(-3).expand(B, H, L, S, E)
        index_sample = torch.randint(S, (L, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L, device=Q.device).unsqueeze(1), index_sample, :]
        
        # 计算QK^T
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # 选择top-k重要性
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # 计算稀疏注意力
        Q_reduce = Q[torch.arange(B, device=Q.device)[:, None, None], 
                     torch.arange(H, device=Q.device)[None, :, None], 
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        
        queries = queries.view(B, L, H, E).permute(0, 2, 1, 3)  # [B, H, L, D]
        keys = keys.view(B, S, H, E).permute(0, 2, 1, 3)  # [B, H, S, D]
        values = values.view(B, S, H, E).permute(0, 2, 1, 3)  # [B, H, S, D]
        
        # 计算稀疏QK
        U_part = self.factor * int(math.ceil(math.log(L)))
        u = self.factor * int(math.ceil(math.log(S)))
        u = min(u, S)
        
        scores_top, index = self._prob_qk(queries, keys, sample_k=U_part, n_top=u)
        
        # 缩放
        scale = self.scale or 1./math.sqrt(E)
        scores_top = scores_top * scale
        
        # Softmax
        attn = torch.softmax(scores_top, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 稀疏上下文
        context = torch.matmul(attn, values)  # [B, H, u, D]
        
        # 恢复形状 - 关键修复：确保输出形状正确
        context_full = torch.zeros(B, H, L, E, device=queries.device)
        context_full[torch.arange(B)[:, None, None], 
                    torch.arange(H)[None, :, None], 
                    index, :] = context
        
        context_full = context_full.permute(0, 2, 1, 3).contiguous()  # [B, L, H, D]
        
        return context_full

# 修正的Informer编码器层
class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = ProbAttention(False, 5, attention_dropout=attention_dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.n_heads = n_heads
        
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # 多头注意力准备
        q = k = v = x
        q = q.view(B, L, self.n_heads, D // self.n_heads)
        k = k.view(B, L, self.n_heads, D // self.n_heads) 
        v = v.view(B, L, self.n_heads, D // self.n_heads)
        
        # 自注意力
        attn_out = self.self_attention(q, k, v)  # [B, L, H, D/H]
        attn_out = attn_out.reshape(B, L, D)  # [B, L, D]
        
        # 残差连接和层归一化
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # 前馈网络
        y = x.transpose(-1, 1)  # [B, D, L]
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(-1, 1)  # [B, L, D]
        
        x = x + y
        x = self.norm2(x)
        
        return x

# 修正的Informer模型
class Informer(nn.Module):
    def __init__(self, num_angles, input_size=2, d_model=64, n_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_angles = num_angles
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 角度特征嵌入
        self.angle_embedding = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = self._positional_encoding(num_angles, d_model)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
        
    def _positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]
        
    def forward(self, x, lengths=None):
        # x: [B, num_angles, 2]
        B = x.shape[0]
        
        # 嵌入角度特征
        embedded = self.angle_embedding(x)  # [B, num_angles, d_model]
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :self.num_angles, :].to(x.device)
        x = embedded + pos_enc
        
        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # [B, d_model]
        
        # 输出预测
        return self.output_layer(x).squeeze(-1)

# 自定义数据集 - 与之前代码相同
class V13Dataset(Dataset):
    def __init__(self, angles, v13_values, lengths, num_layers_angle):
        num_samples, seq_len = angles.shape
        assert seq_len >= num_layers_angle, f"数据中的角度列数({seq_len})小于指定的层数({num_layers_angle})"
        
        angles = angles[:, :num_layers_angle]
        self.num_layers_angle = num_layers_angle
        
        self.angle_features = np.zeros((num_samples, num_layers_angle, 2))
        
        for i in range(num_samples):
            for j in range(num_layers_angle):
                theta = angles[i, j]
                rad = np.radians(theta)
                self.angle_features[i, j, 0] = np.sin(2 * rad)
                self.angle_features[i, j, 1] = np.cos(2 * rad)
        
        self.v13_values = v13_values
        self.lengths = lengths
    
    def __len__(self):
        return len(self.v13_values)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.angle_features[idx], dtype=torch.float32),
            torch.tensor(self.lengths[idx], dtype=torch.int64),
            torch.tensor(self.v13_values[idx], dtype=torch.float32)
        )

# 加载数据函数 - 与之前代码相同
def load_v13_data(file_path, num_layers_angle, max_seq_len=20):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    available_angle_cols = [col for col in df.columns if col.startswith('angle_')]
    available_layers = len(available_angle_cols)
    
    if num_layers_angle > available_layers:
        raise ValueError(
            f"请求的层数 ({num_layers_angle}) 超过数据文件中的可用层数 ({available_layers})\n"
            f"可用角度列: {available_angle_cols}"
        )
    
    print(f"使用前 {num_layers_angle} 层角度数据 (总可用层数: {available_layers})")
    
    angle_columns = [f'angle_{i+1}' for i in range(num_layers_angle)]
    angles = df[angle_columns].values
    v13_values = df['v13'].values
    lengths = np.full(len(angles), num_layers_angle)
    
    print(f"Angle range: {angles.min():.2f}° to {angles.max():.2f}°")
    print(f"v13 range: {v13_values.min():.6f} to {v13_values.max():.6f}")
    
    return angles, v13_values, lengths

# 主程序 - 只保留核心指标
if __name__ == "__main__":
    # 配置参数
    config = {
        'batch_size': 128,
        'd_model': 64,
        'n_heads': 4,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'test_size': 0.2,
        'random_state': 42,
        'num_layers_angle': 16,  # 铺层角度的层数
        'data_file': 'composite_v13_dataset.csv'
    }
    
    # 1. 加载数据
    if not os.path.exists(config['data_file']):
        print(f"Error: Data file {config['data_file']} not found!")
        exit()
    
    angles, v13_values, lengths = load_v13_data(config['data_file'], config['num_layers_angle'])
    
    # 2. v13值归一化
    v13_scaler = StandardScaler()
    v13_scaled = v13_scaler.fit_transform(v13_values.reshape(-1, 1)).flatten()
    
    # 3. 划分数据集
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(angles)), 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    # 4. 创建数据集和加载器
    train_dataset = V13Dataset(angles[train_idx], v13_scaled[train_idx], lengths[train_idx], config['num_layers_angle'])
    test_dataset = V13Dataset(angles[test_idx], v13_scaled[test_idx], lengths[test_idx], config['num_layers_angle'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"铺层角度层数: {config['num_layers_angle']}")
    
    # 5. 初始化Informer模型
    model = Informer(
        num_angles=config['num_layers_angle'],
        input_size=2,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    print("\nInformer Model initialized")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 6. 训练配置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    # 7. 训练模型
    print("\nStarting Informer training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        
        for sequences, lengths_batch, v13_targets in train_loader:
            sequences, v13_targets = sequences.to(device), v13_targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, v13_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths_batch, v13_targets in test_loader:
                sequences, v13_targets = sequences.to(device), v13_targets.to(device)
                lengths_batch = lengths_batch.to(device)
                
                outputs = model(sequences, lengths_batch)
                loss = criterion(outputs, v13_targets)
                val_loss += loss.item() * sequences.size(0)
        
        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # 8. 评估最佳模型 - 只计算核心指标
    model.load_state_dict(best_model_state)
    model.eval()
    
    start_eval_time = time.time()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for sequences, lengths_batch, v13_targets in test_loader:
            sequences, v13_targets = sequences.to(device), v13_targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            outputs = model(sequences, lengths_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(v13_targets.cpu().numpy())
    
    eval_time = time.time() - start_eval_time
    
    # 反归一化
    all_preds_original = v13_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    all_targets_original = v13_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
    
    # 计算核心指标
    r2 = r2_score(all_targets_original, all_preds_original)
    rmse = np.sqrt(mean_squared_error(all_targets_original, all_preds_original))
    
    print("\n=== Informer Evaluation Results ===")
    print(f"Test R² Score: {r2:.4f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Total Computation Time: {training_time + eval_time:.2f} seconds")
    print("==================================")
