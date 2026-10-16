import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import json
import sys

# 导入MCR_Former_alpha模块
# 假设MCR_Former_alpha.py在同一目录下
sys.path.append('.')  # 确保当前目录在Python路径中

# 从MCR_Former_alpha中导入所需的模型
try:
    from MCR_Former_alpha import (
        MCR_Former_Alpha,
        PositionalEncoding,
        MCR_Former_Alpha_2D
    )
    print("Successfully imported models from MCR_Former_alpha.py")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure MCR_Former_alpha.py is in the same directory")
    # 如果导入失败，我们将在后面使用本地的简化版本

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# V13复合材料数据集类 - 支持可变层数
class V13CompositeDataset(Dataset):
    def __init__(self, angles, v13_values, layer_counts, max_layers=30):
        """
        angles: [num_samples, max_layers] - 铺层角度序列，已填充至最大层数
        v13_values: [num_samples] - V13性能值
        layer_counts: [num_samples] - 每个样本的实际层数
        max_layers: 最大层数，用于创建特征张量
        """
        self.layer_counts = layer_counts
        self.max_layers = max_layers
        
        # 创建2D物理感知特征 (sin(2θ), cos(2θ))
        self.layer_features = np.zeros((len(angles), max_layers, 2))
        
        for i in range(len(angles)):
            layers = layer_counts[i]
            for j in range(layers):
                theta = math.radians(angles[i, j])  # 转换为弧度
                
                # 物理感知编码：sin(2θ)/cos(2θ)
                self.layer_features[i, j, 0] = math.sin(2 * theta)
                self.layer_features[i, j, 1] = math.cos(2 * theta)
                
                # 确保[-1, 1]范围
                self.layer_features[i, j, 0] = max(min(self.layer_features[i, j, 0], 1.0), -1.0)
                self.layer_features[i, j, 1] = max(min(self.layer_features[i, j, 1], 1.0), -1.0)
        
        self.v13_values = v13_values
        
    def __len__(self):
        return len(self.v13_values)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.layer_features[idx], dtype=torch.float32),
            torch.tensor(self.layer_counts[idx], dtype=torch.int64),
            torch.tensor(self.v13_values[idx], dtype=torch.float32)
        )

# 加载多个V13复合材料数据集 - 支持不同层数
def load_v13_composite_data(min_layer=1, max_layer=30, data_dir='data'):
    print(f"Loading V13 composite data from layer {min_layer} to {max_layer}...")
    
    all_angles = []
    all_v13_values = []
    all_layer_counts = []
    
    # 第一次遍历找出最大层数
    max_layers_found = 0
    for layer_count in range(min_layer, max_layer + 1):
        filename = os.path.join(data_dir, f'layer_{layer_count}.csv')
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            max_layers_found = max(max_layers_found, layer_count)
            print(f"Found file: layer_{layer_count}.csv with {len(df)} samples")
    
    print(f"Maximum layer count across all files: {max_layers_found}")
    
    # 第二次遍历加载并填充数据
    for layer_count in range(min_layer, max_layer + 1):
        filename = os.path.join(data_dir, f'layer_{layer_count}.csv')
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            angles = df.iloc[:, :-1].values  # 铺层角度
            v13_values = df.iloc[:, -1].values  # V13性能值
            num_samples = len(angles)
            
            # 填充角度数组到最大维度 [num_samples, max_layers]
            padded_angles = np.zeros((num_samples, max_layers_found))
            padded_angles[:, :layer_count] = angles[:, :layer_count]  # 复制实际数据
            
            # 层数信息
            layer_counts = np.full(num_samples, layer_count)
            
            all_angles.append(padded_angles)
            all_v13_values.append(v13_values)
            all_layer_counts.append(layer_counts)
        else:
            print(f"Warning: File {filename} not found, skipping layer {layer_count}")
    
    # 合并所有数据
    angles = np.vstack(all_angles)
    v13_values = np.concatenate(all_v13_values)
    layer_counts = np.concatenate(all_layer_counts)
    
    print(f"\nCombined dataset statistics:")
    print(f"Total samples: {len(angles)}")
    print(f"Layer counts: {np.unique(layer_counts)}")
    print(f"Maximum layer count: {max_layers_found}")
    print(f"Angle array shape after padding: {angles.shape}")
    print(f"Angle range: {angles.min():.2f}° to {angles.max():.2f}°")
    print(f"V13 range: {v13_values.min():.6f} to {v13_values.max():.6f}")
    
    return angles, v13_values, layer_counts, max_layers_found

# 预测函数
def predict_v13(model, angles_tensor, lengths_tensor):
    """预测V13性能值"""
    model.eval()
    with torch.no_grad():
        pred = model(angles_tensor, lengths_tensor).cpu().numpy()
    return pred.flatten()

# 简化版的MCR_Former，如果导入失败则使用此版本
class SimpleMCRFormer(nn.Module):
    """
    简化版的MCR_Former，包含可训练alpha
    """
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_seq_len=30):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 可训练的alpha参数，初始值为0.5
        self.logit_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # sigmoid(0) = 0.5
        
        # 1. 嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 4. 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
    
    def _get_alpha(self):
        """获取当前的alpha值，通过sigmoid确保在[0,1]范围内"""
        return torch.sigmoid(self.logit_alpha)
    
    def get_alpha_value(self):
        """获取当前alpha值（标量）"""
        return self._get_alpha().item()
    
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
        """序列编码 - 共享权重的核心"""
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        
        # 序列聚合 - 加权平均
        if mask is not None:
            valid_mask = (~mask).float().unsqueeze(-1)
            x = torch.sum(x * valid_mask, dim=1) / (torch.sum(valid_mask, dim=1) + 1e-8)
        else:
            x = torch.mean(x, dim=1)
            
        return x
    
    def forward(self, x, lengths=None):
        """前向传播"""
        mask = self._create_mask(x, lengths)
        
        # 路径A：编码原始序列
        feat_forward = self._encode_sequence(x, mask)
        
        # 路径B：编码反转序列
        x_reversed = torch.flip(x, dims=[1])
        feat_backward = self._encode_sequence(x_reversed, mask)
        
        # 获取当前alpha值
        alpha = self._get_alpha()
        
        # 对称池化：加权融合
        fused_feature = alpha * feat_forward + (1 - alpha) * feat_backward
        
        # 回归预测
        return self.regression_head(fused_feature).squeeze(-1)

# JSON序列化辅助函数
def convert_to_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def train_v13_model():
    """训练V13复合材料跨层学习模型"""
    # 配置参数
    config = {
        'batch_size': 64,
        'd_model': 64,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'num_epochs': 60,
        'train_val_split': 0.2,
        'test_size': 0.2,
        'random_state': 42,
        'max_seq_len': 30,
        'min_layer': 1,
        'max_layer': 30,
        'data_dir': 'data',
        'save_prefix': 'v13_mcr_former_alpha',
        'alpha_trainable': True,  # alpha可训练
        'alpha_init': 0.5,  # alpha初始值
    }
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 1. 加载数据
    if not os.path.exists(config['data_dir']):
        raise FileNotFoundError(f"Data directory '{config['data_dir']}' does not exist. Please generate the data first.")
    
    angles, v13_values, layer_counts, max_layers = load_v13_composite_data(
        config['min_layer'], 
        config['max_layer'], 
        config['data_dir']
    )
    total_samples = len(angles)
    
    # 2. 全局shuffle
    rng = np.random.RandomState(config['random_state'])
    shuffled_indices = rng.permutation(total_samples)
    
    # 3. 划分独立测试集
    test_size = int(total_samples * config['test_size'])
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    
    print("\n" + "="*60)
    print(f"DATA SPLITTING STRATEGY - V13 COMPOSITE MATERIALS")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print(f"Training set size: {len(train_indices)} samples (used for model training and validation)")
    print(f"Independent test set size: {len(test_indices)} samples (completely unseen during training)")
    print(f"Training/Validation split ratio: {1-config['train_val_split']:.1f}/{config['train_val_split']:.1f}")
    print("="*60)
    
    # 4. 提取训练集和测试集
    train_angles = angles[train_indices]
    train_v13 = v13_values[train_indices]
    train_layers = layer_counts[train_indices]
    
    test_angles = angles[test_indices]
    test_v13 = v13_values[test_indices]
    test_layers = layer_counts[test_indices]
    
    # 5. 将训练集内部分为训练子集和验证子集
    train_sub_idx, val_sub_idx = train_test_split(
        np.arange(len(train_angles)), 
        test_size=config['train_val_split'], 
        random_state=config['random_state']
    )
    
    # 6. 扩展训练子集，添加反向序列（物理对称性增强）
    print("\nExpanding training subset with reversed sequences (physical symmetry augmentation)...")
    
    # 提取原始训练子集
    train_sub_angles_orig = train_angles[train_sub_idx]
    train_sub_v13_orig = train_v13[train_sub_idx]
    train_sub_layers_orig = train_layers[train_sub_idx]
    
    # 创建反向序列（保持V13值不变，因为物理系统在反转下是对称的）
    train_sub_angles_reversed = np.zeros_like(train_sub_angles_orig)
    for i in range(len(train_sub_angles_orig)):
        layer_count = train_sub_layers_orig[i]
        train_sub_angles_reversed[i, :layer_count] = train_sub_angles_orig[i, :layer_count][::-1]
    
    train_sub_v13_reversed = train_sub_v13_orig.copy()  # V13值保持不变
    train_sub_layers_reversed = train_sub_layers_orig.copy()  # 层数保持不变
    
    # 合并原始训练子集和反向训练子集
    combined_train_angles = np.vstack([train_sub_angles_orig, train_sub_angles_reversed])
    combined_train_v13 = np.hstack([train_sub_v13_orig, train_sub_v13_reversed])
    combined_train_layers = np.hstack([train_sub_layers_orig, train_sub_layers_reversed])
    
    # 验证子集保持不变
    val_sub_angles = train_angles[val_sub_idx]
    val_sub_v13 = train_v13[val_sub_idx]
    val_sub_layers = train_layers[val_sub_idx]
    
    print(f"Original training subset: {len(train_sub_idx)} samples")
    print(f"Expanded training subset: {len(combined_train_angles)} samples (including reversed sequences)")
    print(f"Validation subset: {len(val_sub_idx)} samples")
    print(f"Independent test set: {len(test_indices)} samples")
    
    # 7. 创建数据集和加载器
    train_dataset = V13CompositeDataset(
        combined_train_angles,  # 使用扩展后的训练子集
        combined_train_v13,
        combined_train_layers,
        max_layers
    )
    
    val_dataset = V13CompositeDataset(
        val_sub_angles,
        val_sub_v13,
        val_sub_layers,
        max_layers
    )
    
    test_dataset = V13CompositeDataset(
        test_angles,
        test_v13,
        test_layers,
        max_layers
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"\nDataset sizes after augmentation:")
    print(f"Training subset (after augmentation): {len(train_dataset)} samples")
    print(f"Validation subset: {len(val_dataset)} samples")
    print(f"Independent test set (unseen): {len(test_dataset)} samples")
    print(f"Maximum layers in dataset: {max_layers}")
    
    # 8. 初始化模型 - 使用MCR_Former_alpha中的模型
    print("\nInitializing model from MCR_Former_alpha...")
    
    try:
        # 尝试使用MCR_Former_Alpha_2D，设置alpha可训练
        model = MCR_Former_Alpha_2D(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            alpha=config['alpha_init'],  # 初始值0.5
            alpha_trainable=config['alpha_trainable']  # alpha可训练
        ).to(device)
        print("Successfully initialized MCR_Former_Alpha_2D model")
    except NameError:
        # 如果导入失败，使用简化版本
        print("Using simplified MCR_Former model")
        model = SimpleMCRFormer(
            input_dim=2,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        ).to(device)
    
    # 打印初始alpha值
    initial_alpha = model.get_alpha_value() if hasattr(model, 'get_alpha_value') else 0.5
    print(f"Initial alpha value: {initial_alpha:.4f}")
    
    # 9. 训练模型
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    alpha_history = []  # 记录alpha历史
    
    print("\nStarting training with learnable symmetry parameter (alpha)...")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"Evaluating final performance on {len(test_dataset)} completely unseen samples")
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        batch_alphas = []  # 记录本epoch中每个batch的alpha值
        
        for sequences, lengths_batch, v13_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]", leave=False):
            sequences, v13_targets = sequences.to(device), v13_targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, v13_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防止梯度爆炸
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
            
            # 记录当前alpha值
            if hasattr(model, 'get_alpha_value'):
                current_alpha = model.get_alpha_value()
                batch_alphas.append(current_alpha)
            elif hasattr(model, '_get_alpha'):
                current_alpha = model._get_alpha().item()
                batch_alphas.append(current_alpha)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段（使用验证子集）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths_batch, v13_targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]", leave=False):
                sequences, v13_targets = sequences.to(device), v13_targets.to(device)
                lengths_batch = lengths_batch.to(device)
                
                outputs = model(sequences, lengths_batch)
                loss = criterion(outputs, v13_targets)
                val_loss += loss.item() * sequences.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # 记录平均alpha值
        avg_alpha = np.mean(batch_alphas) if batch_alphas else initial_alpha
        alpha_history.append(float(avg_alpha))  # 转换为Python float
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # 保存alpha值
            if hasattr(model, 'get_alpha_value'):
                learned_alpha = model.get_alpha_value()
            elif hasattr(model, '_get_alpha'):
                learned_alpha = model._get_alpha().item()
            else:
                learned_alpha = avg_alpha
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'max_layers': max_layers,
                'alpha': learned_alpha  # 保存学习到的alpha值
            }, f'results/best_v13_model_{config["save_prefix"]}.pth')
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
              f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
              f'LR: {current_lr:.6f}, Alpha: {avg_alpha:.4f}')
    
    training_time = time.time() - start_time
    print(f"\nTraining completed after {config['num_epochs']} epochs")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f} achieved at epoch {best_epoch+1}")
    
    # 获取最终alpha值
    if hasattr(model, 'get_alpha_value'):
        final_alpha = model.get_alpha_value()
    elif hasattr(model, '_get_alpha'):
        final_alpha = model._get_alpha().item()
    else:
        final_alpha = alpha_history[-1] if alpha_history else initial_alpha
    
    print(f"\nFinal alpha value: {final_alpha:.4f}")
    print(f"Alpha values during training: {alpha_history}")
    
    # 10. 在独立测试集上评估最佳模型
    checkpoint = torch.load(f'results/best_v13_model_{config["save_prefix"]}.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 打印学习到的alpha值
    learned_alpha = checkpoint.get('alpha', final_alpha)
    print(f"\nLearned symmetry parameter (alpha): {learned_alpha:.4f}")
    print(f"Interpretation: {'Nearly symmetric' if abs(learned_alpha-0.5) < 0.05 else 'Adaptively asymmetric'}")
    print(f"  - This indicates the optimal balance between forward and reversed sequences is: "
          f"{learned_alpha:.2%} forward, {(1-learned_alpha):.2%} reversed")
    
    # 独立测试集预测
    all_preds, all_targets = [], []
    all_layers_test = []
    
    with torch.no_grad():
        for sequences, lengths_batch, v13_targets in tqdm(test_loader, desc="Test Evaluation"):
            sequences, v13_targets = sequences.to(device), v13_targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            # 预测
            outputs = model(sequences, lengths_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(v13_targets.cpu().numpy())
            all_layers_test.extend(lengths_batch.cpu().numpy())
    
    # 转换为numpy数组用于计算
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    
    # 计算评估指标
    r2 = r2_score(all_targets_arr, all_preds_arr)
    mse = mean_squared_error(all_targets_arr, all_preds_arr)
    rmse = np.sqrt(mse)
    
    print(f"\nIndependent Test Set Evaluation (completely unseen data):")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Best epoch: {best_epoch+1}")
    print(f"Training samples used: {len(train_sub_idx)} unique samples (before augmentation)")
    print(f"Independent test samples: {len(test_dataset)} samples")
    
    # 11. 保存训练曲线
    plt.figure(figsize=(12, 8))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Alpha值变化曲线
    plt.subplot(2, 2, 2)
    plt.plot(alpha_history, 'r-', linewidth=2)
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Strict Symmetry (alpha=0.5)')
    plt.axhline(y=1.0, color='b', linestyle='--', alpha=0.7, label='Forward-only (alpha=1.0)')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Alpha Value', fontsize=12)
    plt.title('Alpha Value During Training', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.4, 1.05)
    
    # R²散点图
    plt.subplot(2, 2, (3, 4))
    plt.scatter(all_targets_arr, all_preds_arr, alpha=0.7, color='blue', s=30)
    
    # 添加完美预测线
    min_val = min(all_targets_arr.min(), all_preds_arr.min())
    max_val = max(all_targets_arr.max(), all_preds_arr.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.xlabel('Actual V13 Value', fontsize=12)
    plt.ylabel('Predicted V13 Value', fontsize=12)
    plt.title(f'Prediction vs Actual Values (Independent Test Set, R² = {r2:.4f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'results/v13_training_summary_{config["save_prefix"]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 12. 严格对称性验证 - 检查序列反转后的预测一致性（在独立测试集上）
    print("\nValidating physical symmetry preservation on independent test set with learned alpha...")
    symmetry_differences = []
    
    with torch.no_grad():
        for sequences, lengths_batch, _ in test_loader:
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            
            # 原始序列预测
            original_preds = model(sequences, lengths_batch)
            
            # 反转序列预测
            reversed_sequences = torch.flip(sequences, dims=[1])
            reversed_preds = model(reversed_sequences, lengths_batch)
            
            # 计算差异
            diff = torch.abs(original_preds - reversed_preds).cpu().numpy()
            symmetry_differences.extend(diff)
    
    symmetry_differences_arr = np.array(symmetry_differences)
    mean_sym_diff = np.mean(symmetry_differences_arr)
    max_sym_diff = np.max(symmetry_differences_arr)
    perfect_sym_count = np.sum(symmetry_differences_arr < 1e-6)
    perfect_sym_ratio = perfect_sym_count / len(symmetry_differences_arr) * 100
    
    print(f"Learned symmetry validation results (independent test set):")
    print(f"Mean absolute difference (original vs reversed): {mean_sym_diff:.8f}")
    print(f"Maximum absolute difference: {max_sym_diff:.8f}")
    print(f"Samples with high symmetry (<1e-6): {perfect_sym_count}/{len(symmetry_differences_arr)} ({perfect_sym_ratio:.2f}%)")
    print(f"This demonstrates adaptive physical symmetry preservation")
    
    # 13. 按层数分组评估
    print("\nEvaluating performance by layer count...")
    layer_results = {}
    
    # 按层数分组测试数据
    unique_layers = np.unique(test_layers)
    for layer_count in unique_layers:
        indices = np.where(test_layers == layer_count)[0]
        if len(indices) == 0:
            continue
            
        # 提取该层数的预测和真实值
        layer_preds = all_preds_arr[indices]
        layer_targets = all_targets_arr[indices]
        
        # 计算指标
        if np.var(layer_targets) > 1e-8:  # 避免除零错误
            layer_r2 = float(r2_score(layer_targets, layer_preds))
            layer_rmse = float(np.sqrt(mean_squared_error(layer_targets, layer_preds)))
        else:
            layer_r2 = float('nan')
            layer_rmse = float('nan')
        
        layer_results[int(layer_count)] = {
            'r2': layer_r2,
            'rmse': layer_rmse,
            'count': int(len(indices))
        }
        
        print(f"{layer_count} layers: R²={layer_r2:.4f}, RMSE={layer_rmse:.6f}, Samples={len(indices)}")
    
    # 14. 保存结果
    results = {
        'config': config,
        'training_time': float(training_time),
        'best_epoch': int(best_epoch + 1),
        'best_val_loss': float(best_val_loss),
        'initial_alpha': float(initial_alpha),
        'final_alpha': float(final_alpha),
        'learned_alpha': float(learned_alpha),
        'alpha_history': alpha_history,
        'test_results': {
            'r2': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'sample_count': int(len(test_dataset))
        },
        'symmetry_results': {
            'mean_diff': float(mean_sym_diff),
            'max_diff': float(max_sym_diff),
            'high_sym_count': int(perfect_sym_count),
            'high_sym_ratio': float(perfect_sym_ratio),
            'total_samples': int(len(symmetry_differences_arr))
        },
        'layer_results': layer_results
    }
    
    # 将结果转换为可序列化的格式
    results_serializable = convert_to_serializable(results)
    
    # 保存为JSON
    with open(f'results/v13_results_{config["save_prefix"]}.json', 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    # 保存为Excel
    layer_df = pd.DataFrame.from_dict(layer_results, orient='index')
    layer_df.to_excel(f'results/v13_layer_results_{config["save_prefix"]}.xlsx')
    
    # 保存alpha历史
    alpha_df = pd.DataFrame({
        'epoch': range(1, len(alpha_history) + 1),
        'alpha': alpha_history,
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    alpha_df.to_excel(f'results/v13_alpha_history_{config["save_prefix"]}.xlsx', index=False)
    
    print("\n✓ Training, evaluation and symmetry validation completed successfully!")
    print(f"Results saved to 'results' directory with prefix '{config['save_prefix']}'")
    print(f"Key metrics on independent test set: R² = {r2:.4f}, RMSE = {rmse:.6f}, Best Epoch = {best_epoch+1}")
    print(f"Initial alpha: {initial_alpha:.4f}, Final alpha: {final_alpha:.4f}")
    print(f"Physical symmetry preservation: Mean difference = {mean_sym_diff:.8f}, High symmetry ratio = {perfect_sym_ratio:.2f}%")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'initial_alpha': initial_alpha,
        'final_alpha': final_alpha,
        'learned_alpha': learned_alpha,
        'layer_results': layer_results,
        'max_layers': max_layers
    }, f'results/final_v13_model_{config["save_prefix"]}.pth')
    
    print(f"\nFinal model saved to: results/final_v13_model_{config['save_prefix']}.pth")
    
    return model, results

if __name__ == "__main__":
    # 训练V13模型
    model, results = train_v13_model()