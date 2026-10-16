# Baseline for Table 1: self-contained re-implementation of TFT, trained in this work
# on the same 3,000-sequence 16-ply dataset and the same 80/20 split (test_size=0.2,
# random_state=42) as MCR-Former. Original working copy: 07_消融实验/TFT.py.
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

# 修正的TFT核心组件
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.fc(x) * self.sigmoid(self.gate(x))

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        
        # 全局变量选择权重
        self.flattened_size = input_size * num_inputs
        self.variable_selection = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_inputs),
            nn.Softmax(dim=-1)
        )
        
        # 单个变量处理网络
        self.variable_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_inputs)
        ])
    
    def forward(self, x):
        # x: [batch_size, num_inputs, input_size]
        batch_size = x.shape[0]
        
        # Flatten for variable selection
        flattened = x.reshape(batch_size, -1)
        selection_weights = self.variable_selection(flattened)  # [batch_size, num_inputs]
        
        # Process each variable
        processed_vars = []
        for i in range(self.num_inputs):
            var = x[:, i, :]  # [batch_size, input_size]
            processed = self.variable_networks[i](var)  # [batch_size, hidden_size]
            processed_vars.append(processed * selection_weights[:, i:i+1])
        
        # Combine selected variables
        return torch.sum(torch.stack(processed_vars, dim=0), dim=0)  # [batch_size, hidden_size]

class TFTModel(nn.Module):
    def __init__(self, num_angles, input_size=2, hidden_size=64):
        super().__init__()
        self.num_angles = num_angles
        self.hidden_size = hidden_size
        
        # 角度特征处理 - 每个角度输入是2维的(sin(2θ), cos(2θ))
        self.angle_embedding = nn.Linear(input_size, hidden_size)
        
        # 变量选择网络
        self.variable_selection = VariableSelectionNetwork(hidden_size, num_angles, hidden_size)
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention_q = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_k = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_v = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_scale = np.sqrt(hidden_size)
        
        # 输出层 - 修正维度匹配
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # 注意：这里需要hidden_size*2
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x, lengths=None):
        # x: [batch_size, num_angles, 2] - sin(2θ), cos(2θ) features
        batch_size = x.shape[0]
        
        # 处理每个角度特征
        embedded_angles = []
        for i in range(self.num_angles):
            angle_feat = x[:, i, :]  # [batch_size, 2]
            embedded = self.angle_embedding(angle_feat)  # [batch_size, hidden_size]
            embedded_angles.append(embedded.unsqueeze(1))
        
        # Stack to [batch_size, num_angles, hidden_size]
        embedded_seq = torch.cat(embedded_angles, dim=1)
        
        # 变量选择 - 将序列重塑为变量格式
        selected = self.variable_selection(embedded_seq)  # [batch_size, hidden_size]
        
        # LSTM编码
        lstm_out, _ = self.lstm(embedded_seq)  # [batch_size, num_angles, hidden_size*2]
        
        # 注意力机制 - 使用最后一个时间步作为query
        q = self.attention_q(lstm_out[:, -1, :])  # [batch_size, hidden_size]
        k = self.attention_k(lstm_out)  # [batch_size, num_angles, hidden_size]
        v = self.attention_v(lstm_out)  # [batch_size, num_angles, hidden_size]
        
        # 计算注意力分数
        attention_scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)) / self.attention_scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.bmm(attention_weights, v).squeeze(1)  # [batch_size, hidden_size]
        
        # 结合变量选择和注意力输出
        combined = torch.cat([selected, attended], dim=1)  # [batch_size, hidden_size*2]
        
        # 输出预测
        return self.output_layer(combined).squeeze(-1)

# 自定义数据集 - 与原始代码相同
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

# 加载数据函数 - 与原始代码相同
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

# 主程序 - 修正版，只保留核心指标
if __name__ == "__main__":
    # 配置参数
    config = {
        'batch_size': 128,
        'hidden_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 80,
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
    
    # 5. 初始化TFT模型
    model = TFTModel(
        num_angles=config['num_layers_angle'],
        input_size=2,  # sin(2θ), cos(2θ) features
        hidden_size=config['hidden_size']
    ).to(device)
    
    print("\nTemporal Fusion Transformer Model initialized")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 6. 训练配置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    # 7. 训练模型
    print("\nStarting TFT training...")
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
    
    print("\n=== TFT Evaluation Results ===")
    print(f"Test R² Score: {r2:.4f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Total Computation Time: {training_time + eval_time:.2f} seconds")
    print("=============================")
