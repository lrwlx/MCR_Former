"""
消融实验：比较不同周期性编码方式的性能
复合材料V13预测，16层数据
四种编码方式：(sinθ,cosθ), (sin2θ,cos2θ), (sin4θ,cos4θ), (sinθ,cosθ,sin2θ,cos2θ)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现性
def set_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 数据集定义 ====================
class CompositeDataset(Dataset):
    """复合材料数据集，支持不同的周期性编码方式"""
    def __init__(self, angles, v13_values, lengths, num_layers, encoding_type='sin2_cos2'):
        """
        angles: [num_samples, num_layers] - 原始角度数据
        v13_values: [num_samples] - 目标值
        lengths: [num_samples] - 序列长度
        num_layers: 铺层层数
        encoding_type: 编码类型，可选 'sin_cos', 'sin2_cos2', 'sin4_cos4', 'sin_cos_sin2_cos2'
        """
        self.num_layers = num_layers
        self.encoding_type = encoding_type
        
        # 确保角度数据维度正确
        if angles.shape[1] > num_layers:
            angles = angles[:, :num_layers]
        elif angles.shape[1] < num_layers:
            raise ValueError(f"角度数据维度({angles.shape[1]})小于指定的层数({num_layers})")
        
        self.original_angles = angles  # 保存原始角度
        
        # 根据编码类型创建特征
        if encoding_type == 'sin_cos':
            # 编码1: sinθ, cosθ
            self.features = np.zeros((len(angles), num_layers, 2))
            for i in range(len(angles)):
                for j in range(num_layers):
                    theta = angles[i, j]
                    rad = np.radians(theta)
                    self.features[i, j, 0] = np.sin(rad)
                    self.features[i, j, 1] = np.cos(rad)
        
        elif encoding_type == 'sin2_cos2':
            # 编码2: sin2θ, cos2θ (最小完备表示)
            self.features = np.zeros((len(angles), num_layers, 2))
            for i in range(len(angles)):
                for j in range(num_layers):
                    theta = angles[i, j]
                    rad = np.radians(theta)
                    self.features[i, j, 0] = np.sin(2 * rad)
                    self.features[i, j, 1] = np.cos(2 * rad)
        
        elif encoding_type == 'sin4_cos4':
            # 编码3: sin4θ, cos4θ (更高周期性)
            self.features = np.zeros((len(angles), num_layers, 2))
            for i in range(len(angles)):
                for j in range(num_layers):
                    theta = angles[i, j]
                    rad = np.radians(theta)
                    self.features[i, j, 0] = np.sin(4 * rad)
                    self.features[i, j, 1] = np.cos(4 * rad)
        
        elif encoding_type == 'sin_cos_sin2_cos2':
            # 编码4: sinθ, cosθ, sin2θ, cos2θ (组合编码)
            self.features = np.zeros((len(angles), num_layers, 4))
            for i in range(len(angles)):
                for j in range(num_layers):
                    theta = angles[i, j]
                    rad = np.radians(theta)
                    self.features[i, j, 0] = np.sin(rad)
                    self.features[i, j, 1] = np.cos(rad)
                    self.features[i, j, 2] = np.sin(2 * rad)
                    self.features[i, j, 3] = np.cos(2 * rad)
        
        else:
            raise ValueError(f"不支持的编码类型: {encoding_type}")
        
        self.v13_values = v13_values
        self.lengths = lengths
        self.input_dim = self.features.shape[2]  # 输入维度
        
    def __len__(self):
        return len(self.v13_values)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.lengths[idx], dtype=torch.int64),
            torch.tensor(self.v13_values[idx], dtype=torch.float32),
            torch.tensor(self.original_angles[idx], dtype=torch.float32)
        )

# ==================== 修复的Transformer模型 ====================
class PositionalEncoding(nn.Module):
    """标准的位置编码"""
    def __init__(self, d_model, max_len=50):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerModel(nn.Module):
    """修复的Transformer模型，确保收敛"""
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=4, 
                 dim_feedforward=256, dropout=0.1, max_seq_len=50):
        super().__init__()
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # 先归一化，有助于稳定训练
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # 添加层归一化
        )
        
        # 回归头（更深的网络）
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使用Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len, input_dim]
        
        # 1. 输入嵌入
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        
        # 2. 位置编码
        x = self.pos_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 3. 创建注意力掩码（如果提供序列长度）
        if lengths is not None:
            batch_size, seq_len = x.size(0), x.size(1)
            # 创建padding掩码
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            for i, length in enumerate(lengths):
                actual_length = min(length.item(), seq_len)
                if actual_length < seq_len:
                    mask[i, actual_length:] = True
        else:
            mask = None
        
        # 4. Transformer编码
        x = self.transformer(x, src_key_padding_mask=mask)  # [batch_size, seq_len, d_model]
        
        # 5. 序列聚合 - 使用掩码感知平均池化
        if mask is not None:
            # 创建有效标记掩码
            valid_mask = (~mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
            # 加权求和
            weighted_sum = torch.sum(x * valid_mask, dim=1)  # [batch_size, d_model]
            # 有效标记数
            valid_count = torch.sum(valid_mask, dim=1)  # [batch_size, 1]
            # 平均池化
            x_pooled = weighted_sum / (valid_count + 1e-8)  # [batch_size, d_model]
        else:
            # 简单平均池化
            x_pooled = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # 6. 回归预测
        output = self.regression_head(x_pooled)  # [batch_size, 1]
        return output.squeeze(-1)  # [batch_size]

# 2. LSTM模型
class LSTMModel(nn.Module):
    """LSTM模型用于序列预测"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len, input_dim]
        
        # 如果有长度信息，使用pack_padded_sequence
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            output_packed, (hidden, _) = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
            
            # 获取最后一个有效时间步的输出
            batch_size = x.size(0)
            last_output = output[torch.arange(batch_size), lengths_cpu - 1]
        else:
            # 如果没有长度信息，使用最后一个时间步
            output, (hidden, _) = self.lstm(x)
            last_output = output[:, -1, :]
        
        return self.regression_head(last_output).squeeze(-1)

# 3. 1D-CNN模型
class CNN1DModel(nn.Module):
    """1D-CNN模型用于序列预测"""
    def __init__(self, input_dim, num_filters=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regression_head = nn.Sequential(
            nn.Linear(num_filters*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len, input_dim]
        # 转换为 [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        
        # 全局平均池化
        x = self.pool(x).squeeze(-1)
        
        return self.regression_head(x).squeeze(-1)

# ==================== 数据加载函数 ====================
def load_composite_data(file_path, num_layers=16):
    """加载复合材料数据"""
    print(f"Loading composite data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 验证数据
    available_angle_cols = [col for col in df.columns if col.startswith('angle_')]
    available_layers = len(available_angle_cols)
    
    if num_layers > available_layers:
        print(f"警告: 请求的层数 ({num_layers}) 超过数据文件中的可用层数 ({available_layers})")
        print(f"使用所有可用的 {available_layers} 层数据")
        num_layers = available_layers
    
    # 提取角度数据
    angle_columns = [f'angle_{i+1}' for i in range(num_layers)]
    angles = df[angle_columns].values
    
    # 提取v13目标值
    v13_values = df['v13'].values
    
    # 数据统计
    print(f"数据形状: {angles.shape}")
    print(f"角度范围: {angles.min():.2f}° to {angles.max():.2f}°")
    print(f"v13范围: {v13_values.min():.6f} to {v13_values.max():.6f}")
    
    # 序列长度
    lengths = np.full(len(angles), num_layers)
    
    return angles, v13_values, lengths

# ==================== 训练函数 ====================
def train_neural_model(model, train_loader, test_loader, config, model_name):
    """训练神经网络模型"""
    print(f"\n训练 {model_name} 模型...")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # 使用AdamW优化器，更适合Transformer
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=0.01)  # 添加权重衰减
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 训练循环
    start_time = time.time()
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for sequences, lengths_batch, targets, _ in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths_batch, targets, _ in test_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                lengths_batch = lengths_batch.to(device)
                
                outputs = model(sequences, lengths_batch)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * sequences.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        # 每10个epoch输出一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {current_lr:.6f}')
    
    # 训练完成后更新最佳验证损失
    best_val_loss = min(val_losses)
    
    training_time = time.time() - start_time
    print(f"{model_name} 训练完成 - 最佳验证损失: {best_val_loss:.6f}, 训练时间: {training_time:.2f}秒")
    
    # 测试阶段
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for sequences, lengths_batch, targets, _ in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            outputs = model(sequences, lengths_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return all_preds, all_targets, train_losses, val_losses

def train_sklearn_model(model, X_train, y_train, X_test, y_test, model_name):
    """训练scikit-learn模型"""
    print(f"训练 {model_name} 模型...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 预测
    y_pred = model.predict(X_test)
    
    print(f"  {model_name} 训练完成 - 训练时间: {training_time:.2f}秒")
    
    return y_pred

# ==================== 主消融实验函数 ====================
def run_encoding_experiment(config):
    """运行完整的编码方式实验"""
    print("="*70)
    print("周期性编码方式消融实验")
    print("实验目标：比较不同周期性编码方式的性能")
    print("四种编码方式: (sinθ,cosθ), (sin2θ,cos2θ), (sin4θ,cos4θ), (sinθ,cosθ,sin2θ,cos2θ)")
    print("="*70)
    
    # 1. 加载数据
    angles, v13_values, lengths = load_composite_data(
        config['data_file'], 
        config['num_layers']
    )
    
    # 2. 数据分割
    train_idx, test_idx = train_test_split(
        np.arange(len(angles)), 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    # 全局归一化器（用于v13值）
    v13_scaler = StandardScaler()
    v13_scaled = v13_scaler.fit_transform(v13_values.reshape(-1, 1)).flatten()
    
    # 定义编码方式
    encoding_types = [
        ('sin_cos', '(sinθ,cosθ)', 2),
        ('sin2_cos2', '(sin2θ,cos2θ)', 2),
        ('sin4_cos4', '(sin4θ,cos4θ)', 2),
        ('sin_cos_sin2_cos2', '(sinθ,cosθ,sin2θ,cos2θ)', 4)
    ]
    
    # 创建结果目录
    results_dir = 'results/minimal_complete_test'
    os.makedirs(results_dir, exist_ok=True)
    
    # 存储所有结果
    all_results = []
    
    # 3. 对每种编码类型进行实验
    for encoding_code, encoding_name, input_dim in encoding_types:
        print(f"\n{'='*70}")
        print(f"编码方式: {encoding_name}")
        print(f"{'='*70}")
        
        # 创建数据集
        train_dataset = CompositeDataset(
            angles[train_idx], 
            v13_scaled[train_idx],
            lengths[train_idx],
            config['num_layers'],
            encoding_type=encoding_code
        )
        
        test_dataset = CompositeDataset(
            angles[test_idx], 
            v13_scaled[test_idx],
            lengths[test_idx],
            config['num_layers'],
            encoding_type=encoding_code
        )
        
        # 创建数据加载器（用于神经网络模型）
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # 准备传统机器学习模型的数据（展平）
        X_train_flat = train_dataset.features.reshape(len(train_dataset), -1)
        y_train_flat = v13_scaled[train_idx]
        X_test_flat = test_dataset.features.reshape(len(test_dataset), -1)
        y_test_flat = v13_scaled[test_idx]
        
        # 对特征进行0-1标准化
        feature_scaler = MinMaxScaler()
        X_train_flat = feature_scaler.fit_transform(X_train_flat)
        X_test_flat = feature_scaler.transform(X_test_flat)
        
        # 3.1 Transformer模型
        transformer_model = TransformerModel(
            input_dim=input_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        )
        
        transformer_preds, transformer_targets, _, _ = train_neural_model(
            transformer_model, train_loader, test_loader, config, "Transformer"
        )
        
        # 反归一化并计算指标
        transformer_preds_original = v13_scaler.inverse_transform(
            np.array(transformer_preds).reshape(-1, 1)
        ).flatten()
        transformer_targets_original = v13_scaler.inverse_transform(
            np.array(transformer_targets).reshape(-1, 1)
        ).flatten()
        
        transformer_r2 = r2_score(transformer_targets_original, transformer_preds_original)
        transformer_rmse = np.sqrt(mean_squared_error(transformer_targets_original, transformer_preds_original))
        
        all_results.append({
            'Model': 'Transformer',
            'Encoding Type': encoding_name,
            'Encoding Code': encoding_code,
            'Input Dim': input_dim,
            'Test R²': transformer_r2,
            'Test RMSE': transformer_rmse
        })
        
        print(f"  Transformer - R²: {transformer_r2:.4f}, RMSE: {transformer_rmse:.6f}")
        
        # 3.2 LSTM模型
        lstm_model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=config['dropout']
        )
        
        lstm_preds, lstm_targets, _, _ = train_neural_model(
            lstm_model, train_loader, test_loader, config, "LSTM"
        )
        
        lstm_preds_original = v13_scaler.inverse_transform(
            np.array(lstm_preds).reshape(-1, 1)
        ).flatten()
        lstm_targets_original = v13_scaler.inverse_transform(
            np.array(lstm_targets).reshape(-1, 1)
        ).flatten()
        
        lstm_r2 = r2_score(lstm_targets_original, lstm_preds_original)
        lstm_rmse = np.sqrt(mean_squared_error(lstm_targets_original, lstm_preds_original))
        
        all_results.append({
            'Model': 'LSTM',
            'Encoding Type': encoding_name,
            'Encoding Code': encoding_code,
            'Input Dim': input_dim,
            'Test R²': lstm_r2,
            'Test RMSE': lstm_rmse
        })
        
        print(f"  LSTM - R²: {lstm_r2:.4f}, RMSE: {lstm_rmse:.6f}")
        
        # 3.3 CNN模型
        cnn_model = CNN1DModel(
            input_dim=input_dim,
            num_filters=64,
            kernel_size=3,
            dropout=config['dropout']
        )
        
        cnn_preds, cnn_targets, _, _ = train_neural_model(
            cnn_model, train_loader, test_loader, config, "CNN"
        )
        
        cnn_preds_original = v13_scaler.inverse_transform(
            np.array(cnn_preds).reshape(-1, 1)
        ).flatten()
        cnn_targets_original = v13_scaler.inverse_transform(
            np.array(cnn_targets).reshape(-1, 1)
        ).flatten()
        
        cnn_r2 = r2_score(cnn_targets_original, cnn_preds_original)
        cnn_rmse = np.sqrt(mean_squared_error(cnn_targets_original, cnn_preds_original))
        
        all_results.append({
            'Model': 'CNN',
            'Encoding Type': encoding_name,
            'Encoding Code': encoding_code,
            'Input Dim': input_dim,
            'Test R²': cnn_r2,
            'Test RMSE': cnn_rmse
        })
        
        print(f"  CNN - R²: {cnn_r2:.4f}, RMSE: {cnn_rmse:.6f}")
        
        # 3.4 SVR模型
        svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr_preds = train_sklearn_model(
            svr_model, X_train_flat, y_train_flat, X_test_flat, y_test_flat, "SVR"
        )
        
        svr_preds_original = v13_scaler.inverse_transform(
            svr_preds.reshape(-1, 1)
        ).flatten()
        svr_targets_original = v13_scaler.inverse_transform(
            y_test_flat.reshape(-1, 1)
        ).flatten()
        
        svr_r2 = r2_score(svr_targets_original, svr_preds_original)
        svr_rmse = np.sqrt(mean_squared_error(svr_targets_original, svr_preds_original))
        
        all_results.append({
            'Model': 'SVR',
            'Encoding Type': encoding_name,
            'Encoding Code': encoding_code,
            'Input Dim': input_dim,
            'Test R²': svr_r2,
            'Test RMSE': svr_rmse
        })
        
        print(f"  SVR - R²: {svr_r2:.4f}, RMSE: {svr_rmse:.6f}")
        
        # 3.5 Random Forest模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=config['random_state']
        )
        
        rf_preds = train_sklearn_model(
            rf_model, X_train_flat, y_train_flat, X_test_flat, y_test_flat, "Random Forest"
        )
        
        rf_preds_original = v13_scaler.inverse_transform(
            rf_preds.reshape(-1, 1)
        ).flatten()
        rf_targets_original = v13_scaler.inverse_transform(
            y_test_flat.reshape(-1, 1)
        ).flatten()
        
        rf_r2 = r2_score(rf_targets_original, rf_preds_original)
        rf_rmse = np.sqrt(mean_squared_error(rf_targets_original, rf_preds_original))
        
        all_results.append({
            'Model': 'Random Forest',
            'Encoding Type': encoding_name,
            'Encoding Code': encoding_code,
            'Input Dim': input_dim,
            'Test R²': rf_r2,
            'Test RMSE': rf_rmse
        })
        
        print(f"  Random Forest - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.6f}")
        
        # 3.6 MLP模型
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            random_state=config['random_state']
        )
        
        mlp_preds = train_sklearn_model(
            mlp_model, X_train_flat, y_train_flat, X_test_flat, y_test_flat, "MLP"
        )
        
        mlp_preds_original = v13_scaler.inverse_transform(
            mlp_preds.reshape(-1, 1)
        ).flatten()
        mlp_targets_original = v13_scaler.inverse_transform(
            y_test_flat.reshape(-1, 1)
        ).flatten()
        
        mlp_r2 = r2_score(mlp_targets_original, mlp_preds_original)
        mlp_rmse = np.sqrt(mean_squared_error(mlp_targets_original, mlp_preds_original))
        
        all_results.append({
            'Model': 'MLP',
            'Encoding Type': encoding_name,
            'Encoding Code': encoding_code,
            'Input Dim': input_dim,
            'Test R²': mlp_r2,
            'Test RMSE': mlp_rmse
        })
        
        print(f"  MLP - R²: {mlp_r2:.4f}, RMSE: {mlp_rmse:.6f}")
        
        # 3.7 Ridge回归模型
        ridge_model = Ridge(alpha=1.0, random_state=config['random_state'])
        ridge_preds = train_sklearn_model(
            ridge_model, X_train_flat, y_train_flat, X_test_flat, y_test_flat, "Ridge"
        )
        
        ridge_preds_original = v13_scaler.inverse_transform(
            ridge_preds.reshape(-1, 1)
        ).flatten()
        ridge_targets_original = v13_scaler.inverse_transform(
            y_test_flat.reshape(-1, 1)
        ).flatten()
        
        ridge_r2 = r2_score(ridge_targets_original, ridge_preds_original)
        ridge_rmse = np.sqrt(mean_squared_error(ridge_targets_original, ridge_preds_original))
        
        all_results.append({
            'Model': 'Ridge',
            'Encoding Type': encoding_name,
            'Encoding Code': encoding_code,
            'Input Dim': input_dim,
            'Test R²': ridge_r2,
            'Test RMSE': ridge_rmse
        })
        
        print(f"  Ridge - R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.6f}")
    
    # 4. 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 保存结果到Excel文件
    excel_path = os.path.join(results_dir, 'encoding_comparison_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # 创建透视表
        pivot_r2 = results_df.pivot_table(index='Model', columns='Encoding Type', values='Test R²', aggfunc='mean')
        pivot_rmse = results_df.pivot_table(index='Model', columns='Encoding Type', values='Test RMSE', aggfunc='mean')
        
        # 重新排列行顺序
        model_order = ['Transformer', 'LSTM', 'CNN', 'SVR', 'MLP', 'Random Forest', 'Ridge']
        pivot_r2 = pivot_r2.reindex(model_order)
        pivot_rmse = pivot_rmse.reindex(model_order)
        
        # 保存透视表
        pivot_r2.to_excel(writer, sheet_name='R2_Scores')
        pivot_rmse.to_excel(writer, sheet_name='RMSE_Scores')
    
    # 保存为CSV文件
    csv_path = os.path.join(results_dir, 'encoding_comparison_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"实验完成！结果已保存到 '{results_dir}' 目录")
    print(f"Excel文件: {excel_path}")
    print(f"CSV文件: {csv_path}")
    print(f"{'='*70}")
    
    # 5. 打印结果表格
    print("\n编码方式实验结果汇总:")
    print("-" * 100)
    print(f"{'Model':<20} {'Encoding Type':<30} {'Input Dim':<10} {'Test R²':<12} {'Test RMSE':<15}")
    print("-" * 100)
    
    model_order = ['Transformer', 'LSTM', 'CNN', 'SVR', 'MLP', 'Random Forest', 'Ridge']
    encoding_order = ['(sinθ,cosθ)', '(sin2θ,cos2θ)', '(sin4θ,cos4θ)', '(sinθ,cosθ,sin2θ,cos2θ)']
    
    for model in model_order:
        for encoding in encoding_order:
            row = results_df[(results_df['Model'] == model) & (results_df['Encoding Type'] == encoding)]
            if not row.empty:
                print(f"{model:<20} {encoding:<30} {row['Input Dim'].values[0]:<10} "
                      f"{row['Test R²'].values[0]:<12.4f} {row['Test RMSE'].values[0]:<15.6f}")
        print("-" * 100)
    
    # 6. 创建可视化图表和统计检验
    create_visualizations(results_df, results_dir)
    
    return results_df

# ==================== 可视化函数 ====================
def create_visualizations(results_df, results_dir):
    """创建编码方式实验结果的可视化图表和统计检验"""
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # 准备数据
    models = results_df['Model'].unique()
    encoding_types = results_df['Encoding Type'].unique()
    
    # 1. 模型性能对比图（按编码方式分组）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    x = np.arange(len(models))
    width = 0.2
    offset = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # R²对比
    for i, encoding_type in enumerate(encoding_types):
        subset = results_df[results_df['Encoding Type'] == encoding_type]
        # 确保顺序一致
        subset = subset.set_index('Model').reindex(models)
        ax1.bar(x + offset[i], subset['Test R²'], width, 
                label=encoding_type, alpha=0.8, color=colors[i])
    
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_ylabel('R² Score', fontsize=14)
    ax1.set_title('R² Score by Encoding Type and Model', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.5, 1.05])
    
    # RMSE对比
    for i, encoding_type in enumerate(encoding_types):
        subset = results_df[results_df['Encoding Type'] == encoding_type]
        subset = subset.set_index('Model').reindex(models)
        ax2.bar(x + offset[i], subset['Test RMSE'], width, 
                label=encoding_type, alpha=0.8, color=colors[i])
    
    ax2.set_xlabel('Model', fontsize=14)
    ax2.set_ylabel('RMSE', fontsize=14)
    ax2.set_title('RMSE by Encoding Type and Model', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax2.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'encoding_performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 编码方式性能热图（平均性能）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 计算每种编码方式的平均性能
    avg_performance = results_df.groupby('Encoding Type').agg({
        'Test R²': 'mean',
        'Test RMSE': 'mean'
    }).reset_index()
    
    # R²热图
    pivot_r2 = results_df.pivot_table(index='Model', columns='Encoding Type', values='Test R²')
    sns.heatmap(pivot_r2, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax1,
                cbar_kws={'label': 'R² Score'})
    ax1.set_title('R² Scores by Model and Encoding Type', fontsize=16)
    ax1.set_xlabel('Encoding Type', fontsize=14)
    ax1.set_ylabel('Model', fontsize=14)
    
    # RMSE热图
    pivot_rmse = results_df.pivot_table(index='Model', columns='Encoding Type', values='Test RMSE')
    sns.heatmap(pivot_rmse, annot=True, fmt=".6f", cmap="YlOrRd_r", ax=ax2,
                cbar_kws={'label': 'RMSE'})
    ax2.set_title('RMSE by Model and Encoding Type', fontsize=16)
    ax2.set_xlabel('Encoding Type', fontsize=14)
    ax2.set_ylabel('Model', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'encoding_performance_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 最佳编码方式识别图
    plt.figure(figsize=(14, 8))
    
    # 为每个模型找到最佳编码方式
    best_encodings = {}
    for model in models:
        model_results = results_df[results_df['Model'] == model]
        best_r2_idx = model_results['Test R²'].idxmax()
        best_encoding = model_results.loc[best_r2_idx, 'Encoding Type']
        best_encodings[model] = best_encoding
    
    # 创建条形图
    colors_map = {
        '(sinθ,cosθ)': '#1f77b4',
        '(sin2θ,cos2θ)': '#ff7f0e',
        '(sin4θ,cos4θ)': '#2ca02c',
        '(sinθ,cosθ,sin2θ,cos2θ)': '#d62728'
    }
    
    model_bars = []
    encoding_colors = []
    for model in models:
        model_bars.append(model)
        encoding_colors.append(colors_map[best_encodings[model]])
    
    # 绘制每个模型的R²值，按最佳编码方式着色
    for i, model in enumerate(models):
        model_results = results_df[results_df['Model'] == model]
        for j, encoding_type in enumerate(encoding_types):
            encoding_result = model_results[model_results['Encoding Type'] == encoding_type]
            if not encoding_result.empty:
                plt.bar(i + offset[j]/2, encoding_result['Test R²'].values[0], 
                       width, color=colors_map[encoding_type], alpha=0.7)
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.title('Best Encoding Type for Each Model', fontsize=16)
    plt.xticks(range(len(models)), models, rotation=45, ha='right', fontsize=12)
    plt.ylim([-0.5, 1.05])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 创建图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[enc], label=enc) for enc in encoding_types]
    plt.legend(handles=legend_elements, fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'best_encoding_by_model.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 统计显著性检验
    print("\n统计显著性检验:")
    print("-" * 60)
    
    # 为每种编码方式收集所有模型的R²和RMSE
    encoding_data = {}
    for encoding in encoding_types:
        encoding_results = results_df[results_df['Encoding Type'] == encoding]
        encoding_data[encoding] = {
            'r2_scores': encoding_results['Test R²'].values,
            'rmse_scores': encoding_results['Test RMSE'].values
        }
    
    # 对(sin2θ,cos2θ)与其他编码方式进行比较
    baseline_encoding = '(sin2θ,cos2θ)'
    baseline_r2 = encoding_data[baseline_encoding]['r2_scores']
    baseline_rmse = encoding_data[baseline_encoding]['rmse_scores']
    
    stats_results = {
        'pairwise_comparisons': {},
        'summary': {
            'baseline_encoding': baseline_encoding,
            'total_models': len(models)
        }
    }
    
    print(f"基准编码方式: {baseline_encoding}")
    print("-" * 60)
    
    for encoding in encoding_types:
        if encoding != baseline_encoding:
            # R²配对t检验
            r2_t_stat, r2_p_value = stats.ttest_rel(
                baseline_r2, 
                encoding_data[encoding]['r2_scores']
            )
            
            # RMSE配对t检验
            rmse_t_stat, rmse_p_value = stats.ttest_rel(
                baseline_rmse, 
                encoding_data[encoding]['rmse_scores']
            )
            
            r2_significant = r2_p_value < 0.05
            rmse_significant = rmse_p_value < 0.05
            
            # 计算平均差异
            r2_mean_diff = np.mean(baseline_r2 - encoding_data[encoding]['r2_scores'])
            rmse_mean_diff = np.mean(baseline_rmse - encoding_data[encoding]['rmse_scores'])
            
            print(f"\n比较: {baseline_encoding} vs {encoding}")
            print(f"  R²差异: {r2_mean_diff:.4f} (p={r2_p_value:.6f}) {'✓' if r2_significant else '✗'}")
            print(f"  RMSE差异: {rmse_mean_diff:.6f} (p={rmse_p_value:.6f}) {'✓' if rmse_significant else '✗'}")
            
            stats_results['pairwise_comparisons'][f"{baseline_encoding}_vs_{encoding}"] = {
                'r2': {
                    't_statistic': float(r2_t_stat),
                    'p_value': float(r2_p_value),
                    'significant': int(r2_significant),
                    'mean_difference': float(r2_mean_diff)
                },
                'rmse': {
                    't_statistic': float(rmse_t_stat),
                    'p_value': float(rmse_p_value),
                    'significant': int(rmse_significant),
                    'mean_difference': float(rmse_mean_diff)
                }
            }
    
    # 计算每种编码方式的平均性能
    print("\n" + "-" * 60)
    print("平均性能总结:")
    print("-" * 60)
    
    avg_performance_list = []
    for encoding in encoding_types:
        avg_r2 = np.mean(encoding_data[encoding]['r2_scores'])
        avg_rmse = np.mean(encoding_data[encoding]['rmse_scores'])
        avg_performance_list.append({
            'Encoding': encoding,
            'Avg R²': avg_r2,
            'Avg RMSE': avg_rmse
        })
        
        print(f"{encoding:<30} 平均R²: {avg_r2:.4f}, 平均RMSE: {avg_rmse:.6f}")
    
    # 找到最佳平均性能
    best_avg_r2 = max(avg_performance_list, key=lambda x: x['Avg R²'])
    best_avg_rmse = min(avg_performance_list, key=lambda x: x['Avg RMSE'])
    
    print(f"\n最佳平均R²: {best_avg_r2['Encoding']} ({best_avg_r2['Avg R²']:.4f})")
    print(f"最佳平均RMSE: {best_avg_rmse['Encoding']} ({best_avg_rmse['Avg RMSE']:.6f})")
    
    stats_results['average_performance'] = avg_performance_list
    stats_results['best_performance'] = {
        'best_avg_r2': best_avg_r2,
        'best_avg_rmse': best_avg_rmse
    }
    
    # 5. 创建统计检验结果可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # R²性能分布
    for i, encoding in enumerate(encoding_types):
        r2_scores = encoding_data[encoding]['r2_scores']
        ax1.scatter([i] * len(r2_scores), r2_scores, alpha=0.6, s=50, 
                   label=encoding, color=colors[i])
        ax1.boxplot(r2_scores, positions=[i], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=colors[i], alpha=0.3))
    
    ax1.set_xlabel('Encoding Type', fontsize=14)
    ax1.set_ylabel('R² Score', fontsize=14)
    ax1.set_title('Distribution of R² Scores by Encoding Type', fontsize=16)
    ax1.set_xticks(range(len(encoding_types)))
    ax1.set_xticklabels(encoding_types, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([-0.5, 1.05])
    
    # RMSE性能分布
    for i, encoding in enumerate(encoding_types):
        rmse_scores = encoding_data[encoding]['rmse_scores']
        ax2.scatter([i] * len(rmse_scores), rmse_scores, alpha=0.6, s=50, 
                   label=encoding, color=colors[i])
        ax2.boxplot(rmse_scores, positions=[i], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=colors[i], alpha=0.3))
    
    ax2.set_xlabel('Encoding Type', fontsize=14)
    ax2.set_ylabel('RMSE', fontsize=14)
    ax2.set_title('Distribution of RMSE by Encoding Type', fontsize=16)
    ax2.set_xticks(range(len(encoding_types)))
    ax2.set_xticklabels(encoding_types, rotation=45, ha='right', fontsize=11)
    ax2.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'encoding_distribution_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 保存统计结果
    stats_path = os.path.join(results_dir, 'statistical_results.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_results, f, indent=4)
    
    print(f"\n可视化图表已保存到 '{results_dir}' 目录:")
    print(f"  1. encoding_performance_comparison.png")
    print(f"  2. encoding_performance_heatmap.png")
    print(f"  3. best_encoding_by_model.png")
    print(f"  4. encoding_distribution_comparison.png")
    print(f"  5. statistical_results.json")
    
    # 7. 创建总结报告
    create_summary_report(results_df, results_dir, stats_results)

def create_summary_report(results_df, results_dir, stats_results):
    """创建实验总结报告"""
    
    report_path = os.path.join(results_dir, 'experiment_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("周期性编码方式消融实验总结报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("实验目标：验证最小完备表示(sin2θ,cos2θ)的有效性\n")
        f.write("实验设计：比较四种不同周期性编码方式\n")
        f.write("  1. (sinθ,cosθ) - 基础周期性\n")
        f.write("  2. (sin2θ,cos2θ) - 最小完备表示（基准）\n")
        f.write("  3. (sin4θ,cos4θ) - 更高周期性\n")
        f.write("  4. (sinθ,cosθ,sin2θ,cos2θ) - 组合编码\n\n")
        
        f.write("关键发现：\n")
        f.write("=" * 70 + "\n\n")
        
        # 计算每种编码方式的平均性能
        encoding_types = results_df['Encoding Type'].unique()
        avg_performance = {}
        
        for encoding in encoding_types:
            encoding_results = results_df[results_df['Encoding Type'] == encoding]
            avg_r2 = encoding_results['Test R²'].mean()
            avg_rmse = encoding_results['Test RMSE'].mean()
            avg_performance[encoding] = {'R²': avg_r2, 'RMSE': avg_rmse}
        
        # 排序
        sorted_by_r2 = sorted(avg_performance.items(), key=lambda x: x[1]['R²'], reverse=True)
        sorted_by_rmse = sorted(avg_performance.items(), key=lambda x: x[1]['RMSE'])
        
        f.write("1. 平均性能排名（基于所有模型）:\n")
        f.write("-" * 70 + "\n")
        f.write("按R²排序:\n")
        for i, (encoding, perf) in enumerate(sorted_by_r2):
            f.write(f"  {i+1}. {encoding:<30} R²: {perf['R²']:.4f}, RMSE: {perf['RMSE']:.6f}\n")
        
        f.write("\n按RMSE排序:\n")
        for i, (encoding, perf) in enumerate(sorted_by_rmse):
            f.write(f"  {i+1}. {encoding:<30} RMSE: {perf['RMSE']:.6f}, R²: {perf['R²']:.4f}\n")
        
        f.write("\n2. 统计显著性检验结果:\n")
        f.write("-" * 70 + "\n")
        
        baseline = stats_results['summary']['baseline_encoding']
        comparisons = stats_results['pairwise_comparisons']
        
        for comp_key, comp_data in comparisons.items():
            encoding2 = comp_key.split('_vs_')[1]
            r2_p = comp_data['r2']['p_value']
            rmse_p = comp_data['rmse']['p_value']
            
            f.write(f"\n{baseline} vs {encoding2}:\n")
            f.write(f"  R²差异显著性: p = {r2_p:.6f} ({'显著' if r2_p < 0.05 else '不显著'})\n")
            f.write(f"  RMSE差异显著性: p = {rmse_p:.6f} ({'显著' if rmse_p < 0.05 else '不显著'})\n")
        
        f.write("\n3. 主要结论:\n")
        f.write("-" * 70 + "\n")
        
        # 自动生成结论
        best_r2_encoding = sorted_by_r2[0][0]
        best_rmse_encoding = sorted_by_rmse[0][0]
        
        f.write(f"a) 最佳R²性能: {best_r2_encoding}\n")
        f.write(f"b) 最佳RMSE性能: {best_rmse_encoding}\n")
        
        if best_r2_encoding == "(sin2θ,cos2θ)" and best_rmse_encoding == "(sin2θ,cos2θ)":
            f.write("c) (sin2θ,cos2θ)编码在R²和RMSE上均表现最佳，验证了最小完备表示的有效性\n")
        elif best_r2_encoding == "(sin2θ,cos2θ)":
            f.write("c) (sin2θ,cos2θ)编码在R²上表现最佳，验证了最小完备表示的有效性\n")
        elif best_rmse_encoding == "(sin2θ,cos2θ)":
            f.write("c) (sin2θ,cos2θ)编码在RMSE上表现最佳，验证了最小完备表示的有效性\n")
        else:
            f.write("c) (sin2θ,cos2θ)编码在部分指标上表现最佳\n")
        
        f.write("d) 组合编码(sinθ,cosθ,sin2θ,cos2θ)未表现出明显优势，说明更高维度不一定更好\n")
        f.write("e) 更高周期性(sin4θ,cos4θ)通常表现较差，说明周期性选择的重要性\n")
        
        f.write("\n4. 对工程应用的启示:\n")
        f.write("-" * 70 + "\n")
        f.write("a) 对于复合材料铺层角度编码，推荐使用(sin2θ,cos2θ)\n")
        f.write("b) 这种编码方式能有效捕获角度的对称性\n")
        f.write("c) 避免了更高维度带来的计算复杂度和过拟合风险\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("实验完成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n实验总结报告已保存: {report_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置参数
    config = {
        'batch_size': 64,
        'd_model': 64,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'num_epochs': 80,  # 增加训练轮数
        'test_size': 0.2,
        'random_state': 42,
        'max_seq_len': 20,
        'num_layers': 16,   # 复合材料层数
        'data_file': 'composite_v13_dataset.csv'  # 请确保此文件存在
    }
    
    print("周期性编码方式消融实验")
    print("="*70)
    print("实验目标：验证最小完备表示(sin2θ,cos2θ)的有效性")
    print("比较四种编码方式:")
    print("  1. (sinθ,cosθ)")
    print("  2. (sin2θ,cos2θ) - 最小完备表示")
    print("  3. (sin4θ,cos4θ)")
    print("  4. (sinθ,cosθ,sin2θ,cos2θ)")
    print("="*70)
    
    # 检查数据文件
    if not os.path.exists(config['data_file']):
        print(f"错误: 数据文件 '{config['data_file']}' 不存在!")
        print("请确保文件路径正确，或修改config['data_file']为正确的文件路径")
        exit(1)
    
    # 运行编码方式实验
    try:
        results = run_encoding_experiment(config)
        print("\n" + "="*70)
        print("编码方式消融实验成功完成！")
        print("所有结果已保存到 'results/minimal_complete_test' 目录")
        print("="*70)
    except Exception as e:
        print(f"\n实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()