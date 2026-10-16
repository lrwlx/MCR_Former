"""
消融实验：比较不同模型在使用MCR编码和原始角度时的性能
复合材料V13预测，16层数据
修复Transformer版本 - 确保收敛
修改：移除早停，增加RMSE的配对t检验
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
    """复合材料数据集，支持MCR编码和原始角度"""
    def __init__(self, angles, v13_values, lengths, num_layers, use_mcr=True):
        """
        angles: [num_samples, num_layers] - 原始角度数据
        v13_values: [num_samples] - 目标值
        lengths: [num_samples] - 序列长度
        num_layers: 铺层层数
        use_mcr: 是否使用MCR编码
        """
        self.num_layers = num_layers
        self.use_mcr = use_mcr
        
        # 确保角度数据维度正确
        if angles.shape[1] > num_layers:
            angles = angles[:, :num_layers]
        elif angles.shape[1] < num_layers:
            raise ValueError(f"角度数据维度({angles.shape[1]})小于指定的层数({num_layers})")
        
        self.original_angles = angles  # 保存原始角度
        
        if use_mcr:
            # MCR编码: 将角度转换为sin(2θ)和cos(2θ)
            self.features = np.zeros((len(angles), num_layers, 2))
            for i in range(len(angles)):
                for j in range(num_layers):
                    theta = angles[i, j]
                    rad = np.radians(theta)
                    # 最小完备表示: sin(2θ), cos(2θ)
                    self.features[i, j, 0] = np.sin(2 * rad)
                    self.features[i, j, 1] = np.cos(2 * rad)
        else:
            # 原始角度，保留为[num_samples, num_layers, 1]
            self.features = angles.reshape(len(angles), num_layers, 1)
        
        self.v13_values = v13_values
        self.lengths = lengths
        
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
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
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
        # 注意：Transformer期望src_key_padding_mask的形状为[batch_size, seq_len]
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
    # 注释掉早停相关代码
    # patience_counter = 0
    # patience = 15
    
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
        
        # 注释掉早停检查
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        
        # 每5个epoch输出一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {current_lr:.6f}')
        
        # 注释掉早停
        # if patience_counter >= patience:
        #     print(f'早停在epoch {epoch+1}，验证损失不再改善')
        #     break
    
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
def run_ablation_experiment(config):
    """运行完整的消融实验"""
    print("="*70)
    print("消融实验：比较不同模型在使用MCR编码和原始角度时的性能")
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
    global v13_scaler
    v13_scaler = StandardScaler()
    v13_scaled = v13_scaler.fit_transform(v13_values.reshape(-1, 1)).flatten()
    
    # 存储所有结果
    all_results = []
    
    # 3. 对每种特征类型进行实验
    feature_types = ['MCR编码', '原始角度']
    
    for feature_type in feature_types:
        use_mcr = (feature_type == 'MCR编码')
        print(f"\n{'='*70}")
        print(f"特征类型: {feature_type}")
        print(f"{'='*70}")
        
        # 创建数据集
        train_dataset = CompositeDataset(
            angles[train_idx], 
            v13_scaled[train_idx],
            lengths[train_idx],
            config['num_layers'],
            use_mcr=use_mcr
        )
        
        test_dataset = CompositeDataset(
            angles[test_idx], 
            v13_scaled[test_idx],
            lengths[test_idx],
            config['num_layers'],
            use_mcr=use_mcr
        )
        
        # 创建数据加载器（用于神经网络模型）
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # 准备传统机器学习模型的数据（展平）
        X_train_flat = train_dataset.features.reshape(len(train_dataset), -1)
        y_train_flat = v13_scaled[train_idx]
        X_test_flat = test_dataset.features.reshape(len(test_dataset), -1)
        y_test_flat = v13_scaled[test_idx]
        
        # 对原始角度进行0-1标准化（仅当使用原始角度时）
        if not use_mcr:
            feature_scaler = MinMaxScaler()
            X_train_flat = feature_scaler.fit_transform(X_train_flat)
            X_test_flat = feature_scaler.transform(X_test_flat)
        
        # 确定输入维度
        if use_mcr:
            input_dim = 2  # sin(2θ), cos(2θ)
        else:
            input_dim = 1  # 原始角度
        
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
            'Feature Type': feature_type,
            'Test R²': transformer_r2,
            'Test RMSE': transformer_rmse,
            'Input Dim': input_dim
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
            'Feature Type': feature_type,
            'Test R²': lstm_r2,
            'Test RMSE': lstm_rmse,
            'Input Dim': input_dim
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
            'Feature Type': feature_type,
            'Test R²': cnn_r2,
            'Test RMSE': cnn_rmse,
            'Input Dim': input_dim
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
            'Feature Type': feature_type,
            'Test R²': svr_r2,
            'Test RMSE': svr_rmse,
            'Input Dim': input_dim
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
            'Feature Type': feature_type,
            'Test R²': rf_r2,
            'Test RMSE': rf_rmse,
            'Input Dim': input_dim
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
            'Feature Type': feature_type,
            'Test R²': mlp_r2,
            'Test RMSE': mlp_rmse,
            'Input Dim': input_dim
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
            'Feature Type': feature_type,
            'Test R²': ridge_r2,
            'Test RMSE': ridge_rmse,
            'Input Dim': input_dim
        })
        
        print(f"  Ridge - R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.6f}")
    
    # 4. 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 保存结果
    results_filename = 'ablation_results_composite_v13.xlsx'
    with pd.ExcelWriter(results_filename, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # 创建与Table 2格式一致的表格
        pivot_r2 = results_df.pivot(index='Model', columns='Feature Type', values='Test R²')
        pivot_rmse = results_df.pivot(index='Model', columns='Feature Type', values='Test RMSE')
        
        # 重新排列行顺序
        model_order = ['Transformer', 'LSTM', 'CNN', 'SVR', 'MLP', 'Random Forest', 'Ridge']
        pivot_r2 = pivot_r2.reindex(model_order)
        pivot_rmse = pivot_rmse.reindex(model_order)
        
        # 重命名列以匹配Table 2
        pivot_r2.columns = ['raw_angles', 'sin_cos_2theta']
        pivot_rmse.columns = ['raw_angles', 'sin_cos_2theta']
        
        pivot_r2.to_excel(writer, sheet_name='R2_Scores')
        pivot_rmse.to_excel(writer, sheet_name='RMSE_Scores')
    
    print(f"\n{'='*70}")
    print("消融实验完成！")
    print(f"结果已保存到 '{results_filename}'")
    print(f"{'='*70}")
    
    # 5. 打印结果表格（与Table 2格式一致）
    print("\n消融实验结果汇总 (Table 2格式):")
    print("-" * 80)
    print(f"{'Model':<20} {'Feature Type':<20} {'Test R²':<15} {'Test RMSE':<15}")
    print("-" * 80)
    
    for model in model_order:
        # 原始角度
        raw_row = results_df[(results_df['Model'] == model) & (results_df['Feature Type'] == '原始角度')]
        if not raw_row.empty:
            print(f"{model:<20} {'raw_angles':<20} {raw_row['Test R²'].values[0]:<15.4f} {raw_row['Test RMSE'].values[0]:<15.6f}")
        
        # MCR编码
        mcr_row = results_df[(results_df['Model'] == model) & (results_df['Feature Type'] == 'MCR编码')]
        if not mcr_row.empty:
            print(f"{'':<20} {'sin_cos_2theta':<20} {mcr_row['Test R²'].values[0]:<15.4f} {mcr_row['Test RMSE'].values[0]:<15.6f}")
        print("-" * 80)
    
    # 6. 创建可视化图表和统计检验
    create_visualizations(results_df)
    
    return results_df

# ==================== 可视化函数 ====================
def create_visualizations(results_df):
    """创建消融实验结果的可视化图表和统计检验"""
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. 模型性能对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 准备数据
    models = results_df['Model'].unique()
    feature_types = results_df['Feature Type'].unique()
    
    x = np.arange(len(models))
    width = 0.35
    
    # R²对比
    for i, feature_type in enumerate(feature_types):
        subset = results_df[results_df['Feature Type'] == feature_type]
        # 确保顺序一致
        subset = subset.set_index('Model').reindex(models)
        if i == 0:
            ax1.bar(x - width/2, subset['Test R²'], width, label=feature_type, alpha=0.8)
        else:
            ax1.bar(x + width/2, subset['Test R²'], width, label=feature_type, alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance Comparison (R² Score)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.5, 1.05])
    
    # RMSE对比
    for i, feature_type in enumerate(feature_types):
        subset = results_df[results_df['Feature Type'] == feature_type]
        subset = subset.set_index('Model').reindex(models)
        if i == 0:
            ax2.bar(x - width/2, subset['Test RMSE'], width, label=feature_type, alpha=0.8)
        else:
            ax2.bar(x + width/2, subset['Test RMSE'], width, label=feature_type, alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Model Performance Comparison (RMSE)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MCR编码提升效果热图
    plt.figure(figsize=(12, 8))
    
    # 计算MCR编码带来的提升
    model_performance = {}
    for model in models:
        mcr_result = results_df[(results_df['Model'] == model) & (results_df['Feature Type'] == 'MCR编码')]
        raw_result = results_df[(results_df['Model'] == model) & (results_df['Feature Type'] == '原始角度')]
        
        if not mcr_result.empty and not raw_result.empty:
            r2_improvement = mcr_result['Test R²'].values[0] - raw_result['Test R²'].values[0]
            rmse_improvement = raw_result['Test RMSE'].values[0] - mcr_result['Test RMSE'].values[0]  # RMSE降低是好的
            model_performance[model] = {
                'R2_Improvement': r2_improvement,
                'RMSE_Improvement': rmse_improvement
            }
    
    # 创建热图数据
    heatmap_data = pd.DataFrame({
        'R² Improvement': [model_performance[m]['R2_Improvement'] for m in models],
        'RMSE Reduction': [model_performance[m]['RMSE_Improvement'] for m in models]
    }, index=models)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data.T, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
                cbar_kws={'label': 'Performance Improvement'})
    plt.title('MCR Encoding Performance Improvement by Model', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig('mcr_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 统计显著性检验 - 增加RMSE的配对t检验
    print("\n统计显著性检验:")
    print("-" * 50)
    
    # 准备数据
    mcr_r2_scores = results_df[results_df['Feature Type'] == 'MCR编码']['Test R²'].values
    raw_r2_scores = results_df[results_df['Feature Type'] == '原始角度']['Test R²'].values
    mcr_rmse_scores = results_df[results_df['Feature Type'] == 'MCR编码']['Test RMSE'].values
    raw_rmse_scores = results_df[results_df['Feature Type'] == '原始角度']['Test RMSE'].values
    
    # R²配对t检验（同一模型在不同特征类型下的表现）
    r2_t_stat, r2_p_value = stats.ttest_rel(mcr_r2_scores, raw_r2_scores)
    print(f"R²配对t检验结果:")
    print(f"  t-statistic = {r2_t_stat:.4f}")
    print(f"  p-value = {r2_p_value:.6f}")
    
    r2_significant = r2_p_value < 0.05
    print(f"  {'✓' if r2_significant else '✗'} MCR编码与原始角度在R²得分上有显著差异 (p < 0.05)")
    
    # RMSE配对t检验
    rmse_t_stat, rmse_p_value = stats.ttest_rel(raw_rmse_scores, mcr_rmse_scores)  # 注意：我们希望RMSE降低
    print(f"\nRMSE配对t检验结果:")
    print(f"  t-statistic = {rmse_t_stat:.4f}")
    print(f"  p-value = {rmse_p_value:.6f}")
    
    rmse_significant = rmse_p_value < 0.05
    # 由于我们希望MCR编码降低RMSE，所以检查原始角度RMSE是否显著高于MCR编码RMSE
    mean_diff = np.mean(raw_rmse_scores) - np.mean(mcr_rmse_scores)
    if mean_diff > 0:
        direction = "降低"
    else:
        direction = "增加"
    
    print(f"  {'✓' if rmse_significant else '✗'} MCR编码与原始角度在RMSE上有显著差异 (p < 0.05)")
    print(f"  MCR编码使RMSE平均{direction}了{abs(mean_diff):.6f}")
    
    # 计算平均提升
    avg_r2_improvement = np.mean(mcr_r2_scores - raw_r2_scores)
    avg_rmse_improvement = np.mean(raw_rmse_scores - mcr_rmse_scores)
    
    print(f"\n平均性能提升:")
    print(f"  R²平均提升: {avg_r2_improvement:.4f}")
    print(f"  RMSE平均降低: {avg_rmse_improvement:.6f}")
    
    # 保存统计结果（修复JSON序列化错误）
    stats_results = {
        'r2_paired_t_test': {
            't_statistic': float(r2_t_stat),
            'p_value': float(r2_p_value),
            'significant': int(r2_significant),  # 转换为int
            'mean_difference': float(avg_r2_improvement),
            'confidence_interval': list(stats.t.interval(0.95, len(mcr_r2_scores)-1, 
                                                        loc=avg_r2_improvement, 
                                                        scale=stats.sem(mcr_r2_scores - raw_r2_scores)))
        },
        'rmse_paired_t_test': {
            't_statistic': float(rmse_t_stat),
            'p_value': float(rmse_p_value),
            'significant': int(rmse_significant),  # 转换为int
            'mean_difference': float(avg_rmse_improvement),
            'direction': direction,
            'confidence_interval': list(stats.t.interval(0.95, len(mcr_rmse_scores)-1, 
                                                        loc=avg_rmse_improvement, 
                                                        scale=stats.sem(raw_rmse_scores - mcr_rmse_scores)))
        },
        'overall_summary': {
            'avg_r2_improvement': float(avg_r2_improvement),
            'avg_rmse_reduction': float(avg_rmse_improvement),
            'total_models': len(models)
        }
    }
    
    with open('ablation_statistical_results.json', 'w') as f:
        json.dump(stats_results, f, indent=4)
    
    # 4. 创建统计检验结果可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R²提升分布
    r2_improvements = mcr_r2_scores - raw_r2_scores
    ax1.hist(r2_improvements, bins=10, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', label='No Improvement')
    ax1.axvline(x=np.mean(r2_improvements), color='green', linestyle='-', 
                label=f'Mean: {np.mean(r2_improvements):.4f}')
    ax1.set_xlabel('R² Improvement (MCR - Raw)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of R² Improvements', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE降低分布
    rmse_improvements = raw_rmse_scores - mcr_rmse_scores  # 正值表示降低
    ax2.hist(rmse_improvements, bins=10, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', label='No Improvement')
    ax2.axvline(x=np.mean(rmse_improvements), color='green', linestyle='-', 
                label=f'Mean: {np.mean(rmse_improvements):.6f}')
    ax2.set_xlabel('RMSE Reduction (Raw - MCR)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of RMSE Reductions', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('statistical_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化图表已保存:")
    print("  - ablation_performance_comparison.png")
    print("  - mcr_improvement_heatmap.png")
    print("  - statistical_test_results.png")
    print("  - ablation_statistical_results.json")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置参数（使用您提供的超参数）
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
    
    print("复合材料V13预测消融实验")
    print("="*70)
    print("实验目标：比较不同模型在使用MCR编码和原始角度时的性能")
    print("="*70)
    
    # 检查数据文件
    if not os.path.exists(config['data_file']):
        print(f"错误: 数据文件 '{config['data_file']}' 不存在!")
        print("请确保文件路径正确，或修改config['data_file']为正确的文件路径")
        exit(1)
    
    # 运行消融实验
    try:
        results = run_ablation_experiment(config)
        print("\n" + "="*70)
        print("消融实验成功完成！")
        print("="*70)
    except Exception as e:
        print(f"\n实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()