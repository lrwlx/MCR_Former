"""
train_mcr_former_concat.py - 修复版
修复特征相似性分析函数错误
"""

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

# 导入MCR_Former_Concat模型
from MCR_Former_concat import MCR_Former_Concat_2D, MCR_Former_Concat, create_mcr_former_concat

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

# 自定义特征相似性分析函数
def analyze_feature_similarity_local(model, dataloader, device):
    """
    分析前向和后向特征的相似性（适配我们的dataloader返回3个值）
    
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
        for sequences, lengths_batch, _ in dataloader:  # 我们的dataloader返回3个值
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            
            # 获取前向和后向特征
            # 首先需要创建掩码
            mask = model._create_mask(sequences, lengths_batch)
            
            # 获取前向特征
            feat_forward = model._encode_sequence(sequences, mask)
            
            # 获取反向特征
            feat_backward = model._encode_sequence(torch.flip(sequences, dims=[1]), mask)
            
            # 计算余弦相似度
            for f, b in zip(feat_forward, feat_backward):
                cos_sim = torch.nn.functional.cosine_similarity(f.unsqueeze(0), b.unsqueeze(0))
                cosine_similarities.append(cos_sim.item())
                
                # 计算欧几里得距离
                euclidean_dist = torch.norm(f - b, p=2)
                euclidean_distances.append(euclidean_dist.item())
    
    return cosine_similarities, euclidean_distances

# 训练MCR_Former_Concat模型
def train_mcr_former_concat_model():
    """训练MCR_Former_Concat复合材料跨层学习模型"""
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
        'save_prefix': 'mcr_former_concat',
        'model_type': 'concat_2d',  # 可以选择: 'concat', 'concat_2d', 'concat_3d', 'concat_large'
        'use_scheduler': True,
        'weight_decay': 1e-4,
        'grad_clip': 1.0
    }
    
    # 创建结果目录
    os.makedirs('results_mcr_concat', exist_ok=True)
    
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
    print(f"DATA SPLITTING STRATEGY - MCR-Former Concat Model")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print(f"Training set size: {len(train_indices)} samples")
    print(f"Independent test set size: {len(test_indices)} samples")
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
    
    # 8. 初始化MCR_Former_Concat模型
    if config['model_type'] == 'concat_2d':
        model = MCR_Former_Concat_2D(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            norm_first=True
        )
    elif config['model_type'] == 'concat':
        model = MCR_Former_Concat(
            input_dim=2,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            norm_first=True
        )
    elif config['model_type'] == 'concat_large':
        from MCR_Former_concat import MCR_Former_Concat_Large
        model = MCR_Former_Concat_Large(
            input_dim=2,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            norm_first=True
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model = model.to(device)
    
    print(f"\nModel Architecture: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 9. 训练模型
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    if config['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    else:
        scheduler = None
    
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    
    print("\nStarting training with MCR-Former Concat model...")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"Evaluating final performance on {len(test_dataset)} completely unseen samples")
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for sequences, lengths_batch, v13_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]", leave=False):
            sequences, v13_targets = sequences.to(device), v13_targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, v13_targets)
            loss.backward()
            
            # 梯度裁剪
            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
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
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'max_layers': max_layers
            }, f'results_mcr_concat/best_mcr_former_concat_{config["save_prefix"]}.pth')
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {current_lr:.6f}')
    
    training_time = time.time() - start_time
    print(f"\nTraining completed after {config['num_epochs']} epochs")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f} achieved at epoch {best_epoch+1}")
    
    # 10. 在独立测试集上评估最佳模型
    checkpoint = torch.load(f'results_mcr_concat/best_mcr_former_concat_{config["save_prefix"]}.pth', 
                          map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
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
    
    # 11. 对称性验证 - 检查序列反转后的预测一致性
    print("\nValidating physical symmetry preservation on independent test set...")
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
    
    print(f"Symmetry validation results (independent test set):")
    print(f"Mean absolute difference (original vs reversed): {mean_sym_diff:.8f}")
    print(f"Maximum absolute difference: {max_sym_diff:.8f}")
    print(f"Samples with perfect symmetry (<1e-6): {perfect_sym_count}/{len(symmetry_differences_arr)} ({perfect_sym_ratio:.2f}%)")
    
    # 12. 按层数分组评估
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
            layer_r2 = r2_score(layer_targets, layer_preds)
            layer_rmse = np.sqrt(mean_squared_error(layer_targets, layer_preds))
        else:
            layer_r2 = float('nan')
            layer_rmse = float('nan')
        
        # 转换为Python原生类型以支持JSON序列化
        layer_results[int(layer_count)] = {
            'r2': float(layer_r2) if not np.isnan(layer_r2) else None,
            'rmse': float(layer_rmse) if not np.isnan(layer_rmse) else None,
            'count': int(len(indices))
        }
        
        print(f"{layer_count} layers: R²={layer_r2:.4f}, RMSE={layer_rmse:.6f}, Samples={len(indices)}")
    
    # 13. 保存训练曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Loss (MCR-Former Concat)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.plot(learning_rates, linewidth=2, color='red')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.scatter(all_targets_arr, all_preds_arr, alpha=0.7, color='blue', s=30)
    min_val = min(all_targets_arr.min(), all_preds_arr.min())
    max_val = max(all_targets_arr.max(), all_preds_arr.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual V13 Value', fontsize=12)
    plt.ylabel('Predicted V13 Value', fontsize=12)
    plt.title(f'Prediction vs Actual (R² = {r2:.4f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    residuals = all_preds_arr - all_targets_arr
    plt.scatter(all_targets_arr, residuals, alpha=0.7, color='green', s=30)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Actual V13 Value', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'results_mcr_concat/mcr_concat_training_history_{config["save_prefix"]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 14. 保存结果 - 修复JSON序列化错误
    results = {
        'config': config,
        'training_time': float(training_time),
        'best_epoch': int(best_epoch + 1),
        'best_val_loss': float(best_val_loss),
        'test_results': {
            'r2': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'sample_count': int(len(test_dataset))
        },
        'symmetry_results': {
            'mean_diff': float(mean_sym_diff),
            'max_diff': float(max_sym_diff),
            'perfect_sym_count': int(perfect_sym_count),
            'perfect_sym_ratio': float(perfect_sym_ratio),
            'total_samples': int(len(symmetry_differences_arr))
        },
        'layer_results': layer_results
    }
    
    # 保存为JSON - 使用自定义序列化器处理numpy类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(f'results_mcr_concat/mcr_concat_results_{config["save_prefix"]}.json', 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # 保存为Excel
    layer_df = pd.DataFrame.from_dict(layer_results, orient='index')
    layer_df.to_excel(f'results_mcr_concat/mcr_concat_layer_results_{config["save_prefix"]}.xlsx')
    
    # 保存预测结果
    pred_df = pd.DataFrame({
        'actual': all_targets_arr,
        'predicted': all_preds_arr,
        'layer_count': all_layers_test[:len(all_targets_arr)],
        'residual': all_preds_arr - all_targets_arr
    })
    pred_df.to_excel(f'results_mcr_concat/mcr_concat_predictions_{config["save_prefix"]}.xlsx', index=False)
    
    print("\n✓ Training and evaluation completed successfully!")
    print(f"Results saved to 'results_mcr_concat' directory")
    print(f"Key metrics on independent test set: R² = {r2:.4f}, RMSE = {rmse:.6f}")
    print(f"Physical symmetry preservation: Mean difference = {mean_sym_diff:.8f}")
    
    # 15. 特征相似性分析（可选）
    print("\nAnalyzing feature similarity between forward and backward paths...")
    
    try:
        # 使用我们自定义的本地函数
        cosine_sims, euclidean_dists = analyze_feature_similarity_local(model, test_loader, device)
        
        print(f"Mean cosine similarity: {np.mean(cosine_sims):.4f}")
        print(f"Mean Euclidean distance: {np.mean(euclidean_dists):.4f}")
        
        # 保存特征相似性分析结果
        feat_sim_df = pd.DataFrame({
            'cosine_similarity': cosine_sims,
            'euclidean_distance': euclidean_dists
        })
        feat_sim_df.to_excel(f'results_mcr_concat/mcr_concat_feature_similarity_{config["save_prefix"]}.xlsx', index=False)
        
    except Exception as e:
        print(f"Could not analyze feature similarity: {e}")
        print("Skipping feature similarity analysis...")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'layer_results': layer_results,
        'max_layers': int(max_layers),
        'test_results': results['test_results']
    }, f'results_mcr_concat/final_mcr_former_concat_{config["save_prefix"]}.pth')
    
    print(f"\nFinal model saved to: results_mcr_concat/final_mcr_former_concat_{config['save_prefix']}.pth")
    
    return model, results

# 主函数
if __name__ == "__main__":
    # 训练MCR_Former_Concat模型
    print("="*60)
    print("MCR-FORMER CONCAT MODEL TRAINING")
    print("="*60)
    
    model, results = train_mcr_former_concat_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)