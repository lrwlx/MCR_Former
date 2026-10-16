"""
MCR-Former训练与迁移学习脚本（基于MCR_Former_Concat架构）
- 使用1-20层数据训练基础模型
- 在21-25层数据上进行小样本迁移学习（每层10个样本）
- 支持GPU训练
"""

import os
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from tqdm import tqdm

# 从MCR_Former_concat导入模型
from MCR_Former_concat import MCR_Former_Concat_2D

# ==================== 数据集类 ====================
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

# ==================== 数据加载函数 ====================
def load_v13_composite_data(min_layer=1, max_layer=30, data_dir='data'):
    """加载指定层数范围的V13复合材料数据"""
    print(f"\nLoading V13 composite data from layer {min_layer} to {max_layer}...")
    
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
            print(f"  Found file: layer_{layer_count}.csv with {len(df)} samples")
    
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
            print(f"  Warning: File {filename} not found, skipping layer {layer_count}")
    
    # 合并所有数据
    if len(all_angles) == 0:
        raise ValueError(f"No data found for layers {min_layer} to {max_layer}")
    
    angles = np.vstack(all_angles)
    v13_values = np.concatenate(all_v13_values)
    layer_counts = np.concatenate(all_layer_counts)
    
    print(f"\nCombined dataset statistics:")
    print(f"  Total samples: {len(angles)}")
    print(f"  Layer counts: {np.unique(layer_counts)}")
    print(f"  Maximum layer count: {max_layers_found}")
    print(f"  Angle array shape after padding: {angles.shape}")
    print(f"  Angle range: {angles.min():.2f}° to {angles.max():.2f}°")
    print(f"  V13 range: {v13_values.min():.6f} to {v13_values.max():.6f}")
    
    return angles, v13_values, layer_counts, max_layers_found

# ==================== 评估函数 ====================
def evaluate_model(model, dataloader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for sequences, lengths_batch, target_values in dataloader:
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            target_values = target_values.to(device)
            
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, target_values)
            total_loss += loss.item() * sequences.size(0)
            
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target_values.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算评估指标
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    return {
        'loss': avg_loss,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions,
        'targets': targets
    }

# ==================== 训练函数 ====================
def train_model_with_augmentation(model, train_loader, val_loader, config, device, save_path=None):
    """
    训练模型（带数据增强版本）
    参考train_mcr_former_concat_crossLayer_v13.py的训练策略
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # 学习率调度器
    scheduler = None
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    # 梯度裁剪
    grad_clip = config.get('grad_clip', 1.0)
    
    # 训练记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_rmse': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = config.get('early_stop_patience', 15)
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for sequences, lengths_batch, target_values in tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]", 
            leave=False
        ):
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            target_values = target_values.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, target_values)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths_batch, target_values in tqdm(
                val_loader, 
                desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]", 
                leave=False
            ):
                sequences = sequences.to(device)
                lengths_batch = lengths_batch.to(device)
                target_values = target_values.to(device)
                
                outputs = model(sequences, lengths_batch)
                loss = criterion(outputs, target_values)
                val_loss += loss.item() * sequences.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_r2 = val_metrics['r2']
        val_rmse = val_metrics['rmse']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['val_rmse'].append(val_rmse)
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # 保存模型
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_r2': val_r2,
                    'config': config
                }, save_path)
        else:
            patience_counter += 1
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'Val R²: {val_r2:.4f}, LR: {current_lr:.6f}')
        
        # 早停
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    print(f"{'='*80}")
    print(f"Training completed. Best Val Loss: {best_val_loss:.6f}, Best Val R²: {history['val_r2'][np.argmin(history['val_loss'])]:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, training_time

# ==================== 小样本迁移学习函数 ====================
def few_shot_transfer_learning(model, train_loader, val_loader, config, device, 
                               num_epochs=50, lr_multiplier=0.1):
    """
    小样本迁移学习
    - 使用较小的学习率微调
    """
    criterion = nn.MSELoss()
    
    # 使用更小的学习率进行微调
    fine_tune_lr = config['learning_rate'] * lr_multiplier
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=fine_tune_lr,
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_rmse': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"  Fine-tuning for {num_epochs} epochs with LR={fine_tune_lr:.6f}...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for sequences, lengths_batch, target_values in train_loader:
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            target_values = target_values.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, target_values)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_r2 = val_metrics['r2']
        val_rmse = val_metrics['rmse']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['val_rmse'].append(val_rmse)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# ==================== 对称性验证函数 ====================
def validate_symmetry(model, dataloader, device):
    """验证模型对序列反转的对称性保持"""
    model.eval()
    symmetry_differences = []
    
    with torch.no_grad():
        for sequences, lengths_batch, _ in dataloader:
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
    
    return {
        'mean_diff': float(mean_sym_diff),
        'max_diff': float(max_sym_diff),
        'perfect_sym_count': int(perfect_sym_count),
        'perfect_sym_ratio': float(perfect_sym_ratio),
        'total_samples': int(len(symmetry_differences_arr))
    }

# ==================== 主程序 ====================
def main():
    # 配置参数
    config = {
        'batch_size': 64,
        'd_model': 64,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'train_val_split': 0.2,
        'test_size': 0.2,
        'random_state': 42,
        'max_seq_len': 30,
        'min_layer': 1,
        'max_layer': 20,  # 训练使用1-20层
        'data_dir': 'data',
        'save_prefix': 'mcr_former_concat_v13',
        'use_scheduler': True,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'early_stop_patience': 15,
        # 迁移学习参数
        'transfer_num_epochs': 50,
        'transfer_lr_multiplier': 0.1,
        'transfer_few_shot_samples': 10,
    }
    
    # 设置随机种子
    torch.manual_seed(config['random_state'])
    np.random.seed(config['random_state'])
    
    # 设置设备（支持GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # ==================== 步骤1: 加载1-20层训练数据 ====================
    print("\n" + "="*80)
    print("STEP 1: Loading training data (layers 1-20)")
    print("="*80)
    
    train_angles, train_v13, train_layer_counts, max_layers = load_v13_composite_data(
        min_layer=1,
        max_layer=20,
        data_dir=config['data_dir']
    )
    
    total_samples = len(train_angles)
    
    # 全局shuffle
    rng = np.random.RandomState(config['random_state'])
    shuffled_indices = rng.permutation(total_samples)
    
    # 划分独立测试集（从训练数据中划分20%作为独立测试集）
    test_size = int(total_samples * config['test_size'])
    test_indices = shuffled_indices[:test_size]
    train_val_indices = shuffled_indices[test_size:]
    
    print(f"\nData splitting strategy:")
    print(f"  Total samples: {total_samples}")
    print(f"  Training+Validation set: {len(train_val_indices)} samples")
    print(f"  Independent test set: {len(test_indices)} samples")
    
    # 提取训练+验证集和独立测试集
    train_val_angles = train_angles[train_val_indices]
    train_val_v13 = train_v13[train_val_indices]
    train_val_layers = train_layer_counts[train_val_indices]
    
    test_angles = train_angles[test_indices]
    test_v13 = train_v13[test_indices]
    test_layers = train_layer_counts[test_indices]
    
    # 将训练+验证集内部分为训练子集和验证子集
    train_sub_idx, val_sub_idx = train_test_split(
        np.arange(len(train_val_angles)), 
        test_size=config['train_val_split'], 
        random_state=config['random_state']
    )
    
    # 数据增强：添加反向序列
    print("\nExpanding training subset with reversed sequences (physical symmetry augmentation)...")
    
    # 提取原始训练子集
    train_sub_angles_orig = train_val_angles[train_sub_idx]
    train_sub_v13_orig = train_val_v13[train_sub_idx]
    train_sub_layers_orig = train_val_layers[train_sub_idx]
    
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
    val_sub_angles = train_val_angles[val_sub_idx]
    val_sub_v13 = train_val_v13[val_sub_idx]
    val_sub_layers = train_val_layers[val_sub_idx]
    
    print(f"  Original training subset: {len(train_sub_idx)} samples")
    print(f"  Expanded training subset: {len(combined_train_angles)} samples (including reversed sequences)")
    print(f"  Validation subset: {len(val_sub_idx)} samples")
    print(f"  Independent test set: {len(test_indices)} samples")
    
    # 创建数据集和加载器
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
    print(f"  Training subset (after augmentation): {len(train_dataset)} samples")
    print(f"  Validation subset: {len(val_dataset)} samples")
    print(f"  Independent test set (unseen): {len(test_dataset)} samples")
    
    # ==================== 步骤2: 创建并训练模型 ====================
    print("\n" + "="*80)
    print("STEP 2: Creating and training MCR-Former Concat model")
    print("="*80)
    
    model = MCR_Former_Concat_2D(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        norm_first=True  # 提高训练稳定性
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture: {model.__class__.__name__}")
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # 训练模型
    model_save_path = f"models/{config['save_prefix']}_base.pth"
    model, train_history, training_time = train_model_with_augmentation(
        model, 
        train_loader, 
        val_loader, 
        config, 
        device,
        save_path=model_save_path
    )
    
    print(f"\nBase model saved to: {model_save_path}")
    
    # 保存训练历史
    history_save_path = f"results/{config['save_prefix']}_train_history.json"
    with open(history_save_path, 'w') as f:
        serializable_history = {
            k: [float(v) for v in v_list] 
            for k, v_list in train_history.items()
        }
        json.dump(serializable_history, f, indent=2)
    print(f"Training history saved to: {history_save_path}")
    
    # ==================== 步骤3: 在独立测试集上评估 ====================
    print("\n" + "="*80)
    print("STEP 3: Evaluating on independent test set (layers 1-20)")
    print("="*80)
    
    criterion = nn.MSELoss()
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\nIndependent Test Set Evaluation:")
    print(f"  Loss: {test_metrics['loss']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    
    # 对称性验证
    print("\nValidating physical symmetry preservation on independent test set...")
    symmetry_results = validate_symmetry(model, test_loader, device)
    print(f"  Mean absolute difference (original vs reversed): {symmetry_results['mean_diff']:.8f}")
    print(f"  Maximum absolute difference: {symmetry_results['max_diff']:.8f}")
    print(f"  Samples with perfect symmetry (<1e-6): {symmetry_results['perfect_sym_count']}/{symmetry_results['total_samples']} ({symmetry_results['perfect_sym_ratio']:.2f}%)")
    
    # 按层数分组评估
    print("\nEvaluating performance by layer count...")
    layer_results = {}
    unique_layers = np.unique(test_layers)
    
    for layer_count in unique_layers:
        indices = np.where(test_layers == layer_count)[0]
        if len(indices) == 0:
            continue
            
        layer_preds = test_metrics['predictions'][indices]
        layer_targets = test_metrics['targets'][indices]
        
        if np.var(layer_targets) > 1e-8:
            layer_r2 = r2_score(layer_targets, layer_preds)
            layer_rmse = np.sqrt(mean_squared_error(layer_targets, layer_preds))
            layer_mae = mean_absolute_error(layer_targets, layer_preds)
        else:
            layer_r2 = float('nan')
            layer_rmse = float('nan')
            layer_mae = float('nan')
        
        layer_results[int(layer_count)] = {
            'r2': float(layer_r2) if not np.isnan(layer_r2) else None,
            'rmse': float(layer_rmse) if not np.isnan(layer_rmse) else None,
            'mae': float(layer_mae) if not np.isnan(layer_mae) else None,
            'count': int(len(indices))
        }
        
        print(f"  {layer_count} layers: R²={layer_r2:.4f}, RMSE={layer_rmse:.6f}, MAE={layer_mae:.6f}, Samples={len(indices)}")
    
    # ==================== 步骤4: 在21-25层上进行小样本迁移学习 ====================
    print("\n" + "="*80)
    print("STEP 4: Few-shot transfer learning on layers 21-25")
    print("="*80)
    
    transfer_results = {}
    
    for target_layer in range(21, 26):
        print(f"\n{'='*80}")
        print(f"Transfer Learning for Layer {target_layer}")
        print(f"{'='*80}")
        
        # 加载该层数据
        try:
            angles, v13_values, layer_counts, max_layers_target = load_v13_composite_data(
                min_layer=target_layer,
                max_layer=target_layer,
                data_dir=config['data_dir']
            )
        except ValueError as e:
            print(f"  Skipping layer {target_layer}: {e}")
            continue
        
        # 创建数据集
        dataset = V13CompositeDataset(
            angles=angles,
            v13_values=v13_values,
            layer_counts=layer_counts,
            max_layers=max_layers_target
        )
        
        print(f"\n  Total samples for layer {target_layer}: {len(dataset)}")
        
        # 随机选择10个样本作为微调训练集
        num_few_shot = config['transfer_few_shot_samples']
        all_indices = list(range(len(dataset)))
        np.random.shuffle(all_indices)
        
        few_shot_indices = all_indices[:num_few_shot]
        test_indices_layer = all_indices[num_few_shot:]
        
        print(f"  Few-shot training samples: {num_few_shot}")
        print(f"  Test samples: {len(test_indices_layer)}")
        
        # 创建数据子集
        few_shot_dataset = torch.utils.data.Subset(dataset, few_shot_indices)
        test_dataset_layer = torch.utils.data.Subset(dataset, test_indices_layer)
        
        few_shot_loader = DataLoader(
            few_shot_dataset, 
            batch_size=min(num_few_shot, 8),  # 小batch size
            shuffle=True
        )
        test_loader_layer = DataLoader(
            test_dataset_layer, 
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # 创建新模型并加载基础模型权重
        model = MCR_Former_Concat_2D(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            norm_first=True
        ).to(device)
        
        # 加载基础模型权重
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded base model weights")
        
        # 小样本迁移学习
        model, transfer_history = few_shot_transfer_learning(
            model,
            few_shot_loader,
            test_loader_layer,  # 使用测试集作为验证集（因为样本少）
            config,
            device,
            num_epochs=config['transfer_num_epochs'],
            lr_multiplier=config['transfer_lr_multiplier']
        )
        
        # 在测试集上评估
        test_metrics_layer = evaluate_model(model, test_loader_layer, criterion, device)
        
        print(f"\n  Test Results for Layer {target_layer}:")
        print(f"    Loss: {test_metrics_layer['loss']:.6f}")
        print(f"    R²:   {test_metrics_layer['r2']:.4f}")
        print(f"    RMSE: {test_metrics_layer['rmse']:.6f}")
        print(f"    MAE:  {test_metrics_layer['mae']:.6f}")
        
        # 保存结果
        transfer_results[f'layer_{target_layer}'] = {
            'num_samples': len(dataset),
            'few_shot_samples': num_few_shot,
            'test_samples': len(test_indices_layer),
            'metrics': {
                'loss': float(test_metrics_layer['loss']),
                'r2': float(test_metrics_layer['r2']),
                'rmse': float(test_metrics_layer['rmse']),
                'mae': float(test_metrics_layer['mae'])
            }
        }
        
        # 保存迁移学习后的模型
        layer_model_path = f"models/{config['save_prefix']}_layer{target_layer}.pth"
        torch.save({
            'layer': target_layer,
            'model_state_dict': model.state_dict(),
            'metrics': test_metrics_layer
        }, layer_model_path)
        
        # 保存迁移学习历史
        layer_history_path = f"results/{config['save_prefix']}_layer{target_layer}_history.json"
        with open(layer_history_path, 'w') as f:
            serializable_history = {
                k: [float(v) for v in v_list] 
                for k, v_list in transfer_history.items()
            }
            json.dump(serializable_history, f, indent=2)
    
    # ==================== 步骤5: 汇总并保存所有结果 ====================
    print("\n" + "="*80)
    print("STEP 5: Summary of Results")
    print("="*80)
    
    # 保存汇总结果
    summary_path = f"results/{config['save_prefix']}_transfer_summary.json"
    summary_data = {
        'config': config,
        'base_model_path': model_save_path,
        'training_time': float(training_time),
        'base_test_results': {
            'loss': float(test_metrics['loss']),
            'r2': float(test_metrics['r2']),
            'rmse': float(test_metrics['rmse']),
            'mae': float(test_metrics['mae'])
        },
        'symmetry_results': symmetry_results,
        'layer_results': layer_results,
        'transfer_results': transfer_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nTransfer learning results saved to: {summary_path}")
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("BASE MODEL PERFORMANCE (Layers 1-20)")
    print("="*80)
    print(f"{'Layer':<10} {'Samples':<12} {'R²':<12} {'RMSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for layer_num in sorted(layer_results.keys()):
        results = layer_results[layer_num]
        print(f"{layer_num:<10} {results['count']:<12} {results['r2']:<12.4f} "
              f"{results['rmse']:<15.6f} {results['mae']:<15.6f}")
    
    print("\n" + "="*80)
    print("TRANSFER LEARNING RESULTS SUMMARY (Layers 21-25)")
    print("="*80)
    print(f"{'Layer':<10} {'Samples':<12} {'Few-shot':<12} {'R²':<12} {'RMSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for layer_key, results in transfer_results.items():
        layer_num = layer_key.replace('layer_', '')
        metrics = results['metrics']
        print(f"{layer_num:<10} {results['num_samples']:<12} {results['few_shot_samples']:<12} "
              f"{metrics['r2']:<12.4f} {metrics['rmse']:<15.6f} {metrics['mae']:<15.6f}")
    
    print("="*80)
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()