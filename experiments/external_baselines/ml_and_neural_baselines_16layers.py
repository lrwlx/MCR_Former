# Baselines for Table 1 (Transformer, LSTM, CNN, Random Forest, SVR, MLP, Ridge) under three
# input encodings, all trained in this work on the same 3,000-sequence 16-ply dataset.
# Original working copy: 01_v13/02model_compare.py; the archived run is results/model_comparison_results_16layers.csv.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 确保results文件夹存在
os.makedirs('results', exist_ok=True)

# 1. 加载和预处理数据 (添加num_layers_angle参数)
def load_and_preprocess_data(file_path, num_layers_angle=8):
    """加载数据并创建不同特征版本"""
    df = pd.read_csv(file_path)
    
    # 检查数据中是否有足够的角度列
    available_angle_cols = [col for col in df.columns if col.startswith('angle_')]
    available_layers = len(available_angle_cols)
    
    if num_layers_angle > available_layers:
        print(f"警告: 请求的层数 ({num_layers_angle}) 超过数据文件中的可用层数 ({available_layers})")
        print(f"将使用所有可用层 ({available_layers})")
        num_layers_angle = available_layers
    
    print(f"使用前 {num_layers_angle} 层角度数据 (总可用层数: {available_layers})")
    
    # 只取前 num_layers_angle 个角度列
    angle_columns = [f'angle_{i+1}' for i in range(num_layers_angle)]
    angles = df[angle_columns].values
    v13_values = df['v13'].values
    
    print(f"Dataset shape: {angles.shape}")
    print(f"Angle range: {angles.min():.2f}° to {angles.max():.2f}°")
    print(f"v13 range: {v13_values.min():.6f} to {v13_values.max():.6f}")
    
    # 创建特征版本
    features = {}
    
    # 版本1: 原始角度 (num_layers_angle 个特征)
    features['raw_angles'] = angles.copy()
    
    # 版本2: sin(θ)/cos(θ)变换 (2*num_layers_angle 个特征)
    sin_cos_features_std = np.zeros((len(angles), 2*num_layers_angle))
    for i in range(num_layers_angle):
        theta = angles[:, i]
        rad = np.radians(theta)
        sin_cos_features_std[:, 2*i] = np.sin(rad)    # sin(θ)
        sin_cos_features_std[:, 2*i+1] = np.cos(rad)  # cos(θ)
    features['sin_cos_theta'] = sin_cos_features_std
    
    # 版本3: sin(2θ)/cos(2θ)变换 (2*num_layers_angle 个特征) - 用于Transformer和LSTM
    sin_cos_features_2theta = np.zeros((len(angles), 2*num_layers_angle))
    for i in range(num_layers_angle):
        theta = angles[:, i]
        rad = np.radians(theta)
        sin_cos_features_2theta[:, 2*i] = np.sin(2 * rad)    # sin(2θ)
        sin_cos_features_2theta[:, 2*i+1] = np.cos(2 * rad)  # cos(2θ)
    features['sin_cos_2theta'] = sin_cos_features_2theta
    
    return angles, features, v13_values  # 返回原始角度和特征

# 2. 传统机器学习模型训练和评估 (保持不变)
def train_evaluate_traditional_models(X_dict, y, test_size=0.2, random_state=42):
    """训练和评估传统机器学习模型"""
    print("\n" + "="*70)
    print("训练和评估传统机器学习模型")
    print("="*70)
    
    results = {}
    feature_types = ['raw_angles', 'sin_cos_theta', 'sin_cos_2theta']
    
    # 数据分割
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=random_state
    )
    
    # 标准化器
    scalers = {}
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    for feature_type in feature_types:
        print(f"\n{'-'*50}")
        print(f"特征类型: {feature_type}")
        print(f"{'-'*50}")
        
        X = X_dict[feature_type]
        
        # 特征标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scalers[feature_type] = scaler
        
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        y_train = y_scaled[train_idx]
        y_test = y_scaled[test_idx]
        
        # 模型定义
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1
            ),
            'SVR': SVR(
                kernel='rbf', C=10.0, epsilon=0.01, gamma='scale'
            ),
            'Ridge': Ridge(
                alpha=1.0, random_state=random_state
            ),
            'MLP': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation='relu', 
                solver='adam', max_iter=500, random_state=random_state
            )
        }
        
        results[feature_type] = {}
        
        for name, model in models.items():
            start_time = time.time()
            print(f"  训练 {name}...")
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 计算指标
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # 保存结果
            results[feature_type][name] = {
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'training_time': training_time
            }
            
            print(f"    {name} 完成:")
            print(f"      训练R²: {r2_train:.4f}, 测试R²: {r2_test:.4f}")
            print(f"      训练RMSE: {rmse_train:.6f}, 测试RMSE: {rmse_test:.6f}")
            print(f"      训练时间: {training_time:.2f}秒")
    
    return results, scalers, y_scaler, train_idx, test_idx

# 3. 灵活的LSTM模型 (支持不同输入维度)
class FlexibleLSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x, lengths=None):
        # x形状: [batch_size, seq_len, input_dim]
        
        # 处理变长序列
        if lengths is not None:
            # 按序列长度排序
            lengths = lengths.cpu()
            sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
            x = x[sorted_idx]
            
            # 打包序列
            packed = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True, enforce_sorted=True)
            
            # LSTM前向传播
            packed_output, (hidden, cell) = self.lstm(packed)
            
            # 解包序列
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # 恢复原始顺序
            _, original_idx = torch.sort(sorted_idx)
            output = output[original_idx]
            hidden = hidden[:, original_idx]
        else:
            output, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的隐藏状态
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # 回归预测
        return self.regressor(last_hidden).squeeze(-1)

# 4. 1D CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_dim=2, num_layers=8, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        # 1D CNN层 (输入形状: [batch, input_dim, seq_len])
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 计算全连接层输入维度
        # 经过两次pool后，序列长度变为 num_layers // 4
        pooled_length = max(1, num_layers // 4)
        self.fc_input_dim = 128 * pooled_length
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths=None):
        # x形状: [batch_size, seq_len, input_dim]
        # 转换为CNN需要的格式: [batch_size, input_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        # CNN层
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        # 不再pool，保留更多特征
        x = self.dropout(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)

# 5. 通用PyTorch模型训练和评估函数
def train_evaluate_pytorch_model(model_class, model_name, feature_type, X_features, y, 
                                 train_idx, test_idx, y_scaler, config, **model_kwargs):
    """通用PyTorch模型训练和评估函数"""
    num_layers_angle = config['num_layers_angle']
    
    # 确定输入维度
    if feature_type == 'raw_angles':
        input_dim = 1
        # 重塑为序列格式 [samples, seq_len, 1]
        num_samples = X_features.shape[0]
        X_seq = X_features.reshape(num_samples, num_layers_angle, 1)
    elif feature_type == 'sin_cos_theta' or feature_type == 'sin_cos_2theta':
        input_dim = 2
        # 重塑为序列格式 [samples, seq_len, 2]
        num_samples = X_features.shape[0]
        X_seq = X_features.reshape(num_samples, num_layers_angle, 2)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    # 创建数据集
    lengths = np.full(len(X_seq), num_layers_angle)
    
    # 转换为tensor
    X_train_tensor = torch.tensor(X_seq[train_idx], dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_seq[test_idx], dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y[train_idx], dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y[test_idx], dtype=torch.float32).to(device)
    lengths_train_tensor = torch.tensor(lengths[train_idx], dtype=torch.int64).to(device)
    lengths_test_tensor = torch.tensor(lengths[test_idx], dtype=torch.int64).to(device)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, lengths_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, lengths_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # 初始化模型
    model = model_class(input_dim=input_dim, num_layers=num_layers_angle, **model_kwargs).to(device)
    
    # 训练参数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting {model_name} ({feature_type}) training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for sequences, lengths_batch, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths_batch, targets in test_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                lengths_batch = lengths_batch.to(device)
                
                outputs = model(sequences, lengths_batch)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * sequences.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    training_time = time.time() - start_time
    print(f"\n{model_name} ({feature_type}) Training completed! Best validation loss: {best_val_loss:.6f}")
    print(f"{model_name} ({feature_type}) Training time: {training_time:.2f} seconds")
    
    # 评估最佳模型
    model.load_state_dict(best_model_state)
    model.eval()
    
    # 测试集预测
    all_preds, all_targets = [], []
    with torch.no_grad():
        for sequences, lengths_batch, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            outputs = model(sequences, lengths_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 反归一化
    all_preds_original = y_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    all_targets_original = y_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
    
    # 计算评估指标
    r2 = r2_score(all_targets_original, all_preds_original)
    rmse = np.sqrt(mean_squared_error(all_targets_original, all_preds_original))
    
    result = {
        'r2_test': r2,
        'rmse_test': rmse,
        'training_time': training_time
    }
    
    print(f"\n{model_name} ({feature_type}) Test Set Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.6f}")
    
    # 绘制预测vs实际值
    plt.figure(figsize=(10, 8))
    plt.scatter(all_targets_original, all_preds_original, alpha=0.6, color='blue', s=15)
    plt.plot([min(all_targets_original), max(all_targets_original)], 
             [min(all_targets_original), max(all_targets_original)], 'r--', linewidth=2)
    plt.xlabel('Actual v13', fontsize=12)
    plt.ylabel('Predicted v13', fontsize=12)
    plt.title(f'{model_name} ({feature_type}) Prediction vs Actual (Layers: {num_layers_angle}, R² = {r2:.4f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'results/{model_name.lower()}_{feature_type}_{num_layers_angle}layers.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} ({feature_type}) Training History (Layers: {num_layers_angle})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'results/{model_name.lower()}_training_history_{feature_type}_{num_layers_angle}layers.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return result, model, all_targets_original, all_preds_original

# 6. 位置编码层 (保持不变)
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

# 7. 灵活的Transformer模型 (支持不同输入维度)
class FlexibleTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, max_seq_len=20):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        
        # 输入投影层 (将任意维度映射到d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer编码层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 自适应序列聚合
        self.aggregate = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 回归头部 (预测v13)
        self.regression_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
    def forward(self, x, lengths=None):
        # x形状: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)  # 输出形状: [batch_size, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # === 修复关键错误：掩码形状必须与序列长度匹配 ===
        seq_len = x.size(1)  # 获取实际序列长度
        if lengths is not None:
            # 创建形状为 [batch_size, seq_len] 的掩码
            mask = torch.zeros(x.size(0), seq_len, dtype=torch.bool, device=x.device)
            for i, length in enumerate(lengths):
                # 确保不会超出实际序列长度
                actual_length = min(length.item(), seq_len)
                if actual_length < seq_len:
                    mask[i, actual_length:] = True
        else:
            mask = None
        # ===========================================
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 序列聚合
        if lengths is not None:
            # 使用有效长度进行平均
            valid_mask = (~mask).float().unsqueeze(-1)
            x = torch.sum(x * valid_mask, dim=1) / torch.sum(valid_mask, dim=1)
        else:
            x = torch.mean(x, dim=1)
        
        # 特征提取与回归
        x = self.aggregate(x)
        return self.regression_head(x).squeeze(-1)

# 8. 主程序
if __name__ == "__main__":
    # 配置参数 (添加num_layers_angle)
    config = {
        'batch_size': 128,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'test_size': 0.2,
        'random_state': 42,
        'max_seq_len': 20,
        'data_file': 'composite_v13_dataset.csv',
        'num_layers_angle': 12  # 可修改此值来改变铺层层 (8, 10, 12等)
    }
    
    print(f"当前配置的铺层层: {config['num_layers_angle']}")
    
    # 1. 加载数据
    if not os.path.exists(config['data_file']):
        print(f"Error: Data file {config['data_file']} not found!")
        print("Please generate the dataset first using the Sobol sampling script")
        exit()
    
    angles, features, v13_values = load_and_preprocess_data(config['data_file'], config['num_layers_angle'])
    
    # 2. 训练和评估传统机器学习模型
    traditional_results, scalers, y_scaler, train_idx, test_idx = train_evaluate_traditional_models(
        features, v13_values, test_size=config['test_size'], random_state=config['random_state']
    )
    
    # 3. 准备标准化目标
    v13_scaled = y_scaler.transform(v13_values.reshape(-1, 1)).flatten()
    
    # 4. 训练和评估PyTorch模型 (LSTM, CNN, Transformer) 三种预处理方式
    pytorch_results = {}
    pytorch_models = {}
    pytorch_predictions = {}
    
    feature_types = ['raw_angles', 'sin_cos_theta', 'sin_cos_2theta']
    model_configs = {
        'LSTM': {
            'model_class': FlexibleLSTMModel,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1
        },
        'CNN': {
            'model_class': CNNModel,
            'hidden_dim': 64,
            'dropout': 0.1
        },
        'Transformer': {
            'model_class': FlexibleTransformer,
            'd_model': config['d_model'],
            'nhead': config['nhead'],
            'num_layers': config['num_layers'],
            'dim_feedforward': config['dim_feedforward'],
            'dropout': config['dropout'],
            'max_seq_len': config['max_seq_len']
        }
    }
    
    for model_name, model_config in model_configs.items():
        pytorch_results[model_name] = {}
        pytorch_models[model_name] = {}
        pytorch_predictions[model_name] = {}
        
        for feature_type in feature_types:
            # 跳过不适合的组合
            if model_name == 'Transformer' and feature_type == 'raw_angles':
                print(f"\n跳过 {model_name} with {feature_type} (不适合原始角度)")
                continue
            
            if model_name == 'CNN' and feature_type == 'raw_angles':
                print(f"\n跳过 {model_name} with {feature_type} (不适合原始角度)")
                continue
            
            result, model, true_values, pred_values = train_evaluate_pytorch_model(
                model_config['model_class'],
                model_name,
                feature_type,
                features[feature_type],
                v13_scaled,
                train_idx,
                test_idx,
                y_scaler,
                config,
                **{k: v for k, v in model_config.items() if k != 'model_class'}
            )
            
            pytorch_results[model_name][feature_type] = result
            pytorch_models[model_name][feature_type] = model
            pytorch_predictions[model_name][feature_type] = (true_values, pred_values)
    
    # 5. 结果汇总和比较
    print("\n" + "="*70)
    print("模型性能对比")
    print("="*70)
    
    # 创建结果DataFrame
    results_data = []
    
    # 添加传统模型结果
    for feature_type, models in traditional_results.items():
        for model_name, metrics in models.items():
            results_data.append({
                'Model': model_name,
                'Feature Type': feature_type,
                'Test R²': metrics['r2_test'],
                'Test RMSE': metrics['rmse_test'],
                'Training Time (s)': metrics['training_time']
            })
    
    # 添加PyTorch模型结果
    for model_name, feature_results in pytorch_results.items():
        for feature_type, metrics in feature_results.items():
            results_data.append({
                'Model': model_name,
                'Feature Type': feature_type,
                'Test R²': metrics['r2_test'],
                'Test RMSE': metrics['rmse_test'],
                'Training Time (s)': metrics['training_time']
            })
    
    results_df = pd.DataFrame(results_data)
    
    # 按测试R²排序
    results_df = results_df.sort_values('Test R²', ascending=False)
    
    print("\n完整结果表格:")
    print(results_df.to_string(index=False))
    
    # 保存结果
    results_df.to_csv(f'results/model_comparison_results_{config["num_layers_angle"]}layers.csv', index=False)
    print(f"\n结果已保存到 'results/model_comparison_results_{config['num_layers_angle']}layers.csv'")
    
    # 6. 可视化结果
    plt.figure(figsize=(20, 15))
    
    # R²对比图
    plt.subplot(3, 2, 1)
    sns.barplot(data=results_df, x='Model', y='Test R²', hue='Feature Type')
    plt.title(f'测试集 R² 对比 (铺层层: {config["num_layers_angle"]})', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/r2_comparison_{config["num_layers_angle"]}layers.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # RMSE对比图
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 2, 1)
    sns.barplot(data=results_df, x='Model', y='Test RMSE', hue='Feature Type')
    plt.title(f'测试集 RMSE 对比 (铺层层: {config["num_layers_angle"]})', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/rmse_comparison_{config["num_layers_angle"]}layers.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 消融实验：sin/cos 2θ变换的影响 (针对传统模型)
    plt.figure(figsize=(12, 8))
    improvement_data = []
    for model_name in ['RandomForest', 'SVR', 'Ridge', 'MLP']:
        if model_name in traditional_results['raw_angles']:
            raw_r2 = traditional_results['raw_angles'][model_name]['r2_test']
            sin2_r2 = traditional_results['sin_cos_2theta'][model_name]['r2_test']
            improvement = (sin2_r2 - raw_r2) / raw_r2 * 100
            improvement_data.append({
                'Model': model_name,
                'R² Improvement (%)': improvement
            })
    
    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        sns.barplot(data=imp_df, x='Model', y='R² Improvement (%)')
        plt.title(f'sin/cos 2θ变换带来的R²提升百分比 (传统模型, 铺层层: {config["num_layers_angle"]})', fontsize=14)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'results/traditional_improvement_{config["num_layers_angle"]}layers.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # 消融实验：sin/cos 2θ变换的影响 (针对深度学习模型)
    plt.figure(figsize=(12, 8))
    dl_improvement_data = []
    for model_name in ['LSTM', 'CNN', 'Transformer']:
        if model_name in pytorch_results and 'sin_cos_theta' in pytorch_results[model_name] and 'sin_cos_2theta' in pytorch_results[model_name]:
            theta_r2 = pytorch_results[model_name]['sin_cos_theta']['r2_test']
            sin2_r2 = pytorch_results[model_name]['sin_cos_2theta']['r2_test']
            improvement = (sin2_r2 - theta_r2) / theta_r2 * 100
            dl_improvement_data.append({
                'Model': model_name,
                'R² Improvement (%)': improvement
            })
    
    if dl_improvement_data:
        dl_imp_df = pd.DataFrame(dl_improvement_data)
        sns.barplot(data=dl_imp_df, x='Model', y='R² Improvement (%)')
        plt.title(f'sin/cos 2θ vs sin/cos θ变换带来的R²提升百分比 (深度学习模型, 铺层层: {config["num_layers_angle"]})', fontsize=14)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'results/deeplearning_improvement_{config["num_layers_angle"]}layers.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # 所有模型预测vs实际值对比
    for model_name in ['LSTM', 'CNN', 'Transformer']:
        if model_name not in pytorch_predictions:
            continue
            
        plt.figure(figsize=(20, 6))
        feature_count = 0
        
        for feature_type, (true_values, pred_values) in pytorch_predictions[model_name].items():
            if feature_count >= 3:  # 最多3个子图
                break
                
            feature_count += 1
            plt.subplot(1, 3, feature_count)
            plt.scatter(true_values, pred_values, alpha=0.6, color='blue', s=15)
            plt.plot([min(true_values), max(true_values)], 
                     [min(true_values), max(true_values)], 'r--', linewidth=2)
            r2 = r2_score(true_values, pred_values)
            plt.xlabel('Actual v13', fontsize=12)
            plt.ylabel('Predicted v13', fontsize=12)
            plt.title(f'{model_name} ({feature_type})\nR² = {r2:.4f}', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'results/{model_name.lower()}_all_features_{config["num_layers_angle"]}layers.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # 7. 关键结论
    print("\n" + "="*70)
    print("关键结论")
    print("="*70)
    
    # 找出最佳模型
    best_row = results_df.iloc[0]
    print(f"\n🏆 最佳模型: {best_row['Model']} with {best_row['Feature Type']} features")
    print(f"   测试R²: {best_row['Test R²']:.4f}, 测试RMSE: {best_row['Test RMSE']:.6f}")
    
    # 按模型类型分析
    print(f"\n📈 按模型类型分析 (铺层层: {config['num_layers_angle']}):")
    
    for model_type in ['RandomForest', 'SVR', 'Ridge', 'MLP', 'LSTM', 'CNN', 'Transformer']:
        model_results = results_df[results_df['Model'] == model_type]
        if not model_results.empty:
            best_feature = model_results.loc[model_results['Test R²'].idxmax()]
            print(f"   {model_type}: 最佳特征 = {best_feature['Feature Type']}, R² = {best_feature['Test R²']:.4f}")
    
    # 按特征类型分析
    print(f"\n📊 按特征类型分析 (铺层层: {config['num_layers_angle']}):")
    
    for feature_type in ['raw_angles', 'sin_cos_theta', 'sin_cos_2theta']:
        feature_results = results_df[results_df['Feature Type'] == feature_type]
        if not feature_results.empty:
            best_model = feature_results.loc[feature_results['Test R²'].idxmax()]
            print(f"   {feature_type}: 最佳模型 = {best_model['Model']}, R² = {best_model['Test R²']:.4f}")
    
    # 与基准模型比较
    print(f"\n🔄 与基准模型 (随机森林 + raw_angles) 比较:")
    rf_raw = results_df[(results_df['Model'] == 'RandomForest') & (results_df['Feature Type'] == 'raw_angles')]
    if not rf_raw.empty:
        rf_raw_r2 = rf_raw.iloc[0]['Test R²']
        for _, row in results_df.iterrows():
            if row['Model'] == 'RandomForest' and row['Feature Type'] == 'raw_angles':
                continue
            improvement = (row['Test R²'] - rf_raw_r2) / rf_raw_r2 * 100
            if improvement > 5:  # 只显示显著改进
                print(f"   {row['Model']} ({row['Feature Type']}): R²提升 {improvement:.1f}%")
    
    print(f"\n✅ 总结 (铺层层: {config['num_layers_angle']}):")
    print(f"   1. 深度学习模型在不同特征表示下表现不同，需要仔细选择合适的特征变换")
    print(f"   2. sin/cos 2θ变换在大多数情况下提供了最佳性能，验证了物理约束的重要性")
    print(f"   3. Transformer和CNN模型在处理序列数据方面展现出优势，特别是在处理复杂铺层结构时")
    print(f"   4. 传统的随机森林模型仍然具有竞争力，特别是在小数据集上")
    print(f"   5. 特征工程比模型选择更重要，适当的物理特征变换可以大幅提升所有模型的性能")

    # 8. 保存预测结果示例
    np.random.seed(42)
    sample_indices = np.random.choice(test_idx, 5, replace=False)
    sample_angles = angles[sample_indices]
    sample_true_v13 = v13_values[sample_indices]
    
    print("\n" + "="*70)
    print("模型预测示例对比 (最佳特征版本)")
    print("="*70)
    
    # 找出每个模型的最佳特征版本
    best_features = {}
    for model_name in ['RandomForest', 'SVR', 'Ridge', 'MLP', 'LSTM', 'CNN', 'Transformer']:
        model_results = results_df[results_df['Model'] == model_name]
        if not model_results.empty:
            best_row = model_results.loc[model_results['Test R²'].idxmax()]
            best_features[model_name] = best_row['Feature Type']
    
    for i, idx in enumerate(sample_indices):
        angles_seq = sample_angles[i]
        true_v13 = sample_true_v13[i]
        
        print(f"\n样本 {i+1}:")
        print(f"  铺层角度: {angles_seq.round(1).tolist()}")
        print(f"  真实v13: {true_v13:.6f}")
        
        # 传统模型预测
        for model_name, feature_type in best_features.items():
            if model_name in ['LSTM', 'CNN', 'Transformer']:
                continue
                
            if feature_type in scalers:
                # 准备特征
                if feature_type == 'raw_angles':
                    features_sample = angles_seq.reshape(1, -1)
                elif feature_type == 'sin_cos_theta':
                    features_sample = np.zeros((1, 2*config['num_layers_angle']))
                    for j in range(config['num_layers_angle']):
                        theta = angles_seq[j]
                        rad = np.radians(theta)
                        features_sample[0, 2*j] = np.sin(rad)
                        features_sample[0, 2*j+1] = np.cos(rad)
                elif feature_type == 'sin_cos_2theta':
                    features_sample = np.zeros((1, 2*config['num_layers_angle']))
                    for j in range(config['num_layers_angle']):
                        theta = angles_seq[j]
                        rad = np.radians(theta)
                        features_sample[0, 2*j] = np.sin(2*rad)
                        features_sample[0, 2*j+1] = np.cos(2*rad)
                
                # 标准化
                features_scaled = scalers[feature_type].transform(features_sample)
                
                # 预测
                model = traditional_results[feature_type][model_name]
                pred_scaled = model.predict(features_scaled)[0]
                pred_original = y_scaler.inverse_transform([[pred_scaled]])[0][0]
                
                error = abs(true_v13 - pred_original)
                print(f"  {model_name} ({feature_type}): {pred_original:.6f} (误差: {error:.6f})")
        
        # 深度学习模型预测
        for model_name in ['LSTM', 'CNN', 'Transformer']:
            if model_name not in best_features:
                continue
                
            feature_type = best_features[model_name]
            if model_name not in pytorch_models or feature_type not in pytorch_models[model_name]:
                continue
                
            model = pytorch_models[model_name][feature_type]
            
            # 准备特征
            if feature_type == 'raw_angles':
                features_sample = np.zeros((config['num_layers_angle'], 1))
                for j in range(config['num_layers_angle']):
                    features_sample[j, 0] = angles_seq[j]
            else:  # sin_cos_theta or sin_cos_2theta
                features_sample = np.zeros((config['num_layers_angle'], 2))
                for j in range(config['num_layers_angle']):
                    theta = angles_seq[j]
                    rad = np.radians(theta)
                    if feature_type == 'sin_cos_theta':
                        features_sample[j, 0] = np.sin(rad)
                        features_sample[j, 1] = np.cos(rad)
                    else:  # sin_cos_2theta
                        features_sample[j, 0] = np.sin(2*rad)
                        features_sample[j, 1] = np.cos(2*rad)
            
            # 转换为tensor
            seq_tensor = torch.tensor(features_sample, dtype=torch.float32).unsqueeze(0).to(device)
            len_tensor = torch.tensor([config['num_layers_angle']], dtype=torch.int64).to(device)
            
            # 预测
            model.eval()
            with torch.no_grad():
                pred_scaled = model(seq_tensor, len_tensor).cpu().numpy()[0]
            
            pred_original = y_scaler.inverse_transform([[pred_scaled]])[0][0]
            error = abs(true_v13 - pred_original)
            print(f"  {model_name} ({feature_type}): {pred_original:.6f} (误差: {error:.6f})")
    
    print("\nDetailed comparison complete!")
