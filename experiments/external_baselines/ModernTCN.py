# Baseline for Table 1: self-contained re-implementation of ModernTCN, trained in this work
# on the same 3,000-sequence 16-ply dataset and the same 80/20 split (test_size=0.2,
# random_state=42) as MCR-Former. Original working copy: 07_消融实验/ModernTCN.py.
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

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ModernTCN核心组件 - 简化版
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        # x: [B, C, L]
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = x + residual  # 残差连接
        x = self.relu(x)
        return x

class ModernTCN(nn.Module):
    def __init__(self, num_angles, input_size=2, hidden_size=64, num_blocks=3, dropout=0.1):
        super().__init__()
        self.num_angles = num_angles
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # TCN块
        self.blocks = nn.ModuleList()
        current_channels = hidden_size
        for i in range(num_blocks):
            dilation = 2 ** i  # 指数增长的膨胀率
            block = ResidualBlock(
                in_channels=current_channels,
                out_channels=hidden_size,
                kernel_size=3,
                dilation=dilation,
                dropout=dropout
            )
            self.blocks.append(block)
            current_channels = hidden_size
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x, lengths=None):
        # x: [B, num_angles, 2]
        # 嵌入输入特征
        x = self.input_embedding(x)  # [B, num_angles, hidden_size]
        
        # 转换为卷积格式 [B, C, L]
        x = x.permute(0, 2, 1)
        
        # 通过TCN块
        for block in self.blocks:
            x = block(x)
        
        # 全局平均池化
        x = self.global_pool(x)  # [B, hidden_size, 1]
        x = x.squeeze(-1)  # [B, hidden_size]
        
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
        'hidden_size': 64,
        'num_blocks': 3,
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
    
    # 5. 初始化ModernTCN模型
    model = ModernTCN(
        num_angles=config['num_layers_angle'],
        input_size=2,
        hidden_size=config['hidden_size'],
        num_blocks=config['num_blocks'],
        dropout=config['dropout']
    ).to(device)
    
    print("\nModernTCN Model initialized")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 6. 训练配置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    # 7. 训练模型
    print("\nStarting ModernTCN training...")
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
    
    print("\n=== ModernTCN Evaluation Results ===")
    print(f"Test R² Score: {r2:.4f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Total Computation Time: {training_time + eval_time:.2f} seconds")
    print("==================================")
