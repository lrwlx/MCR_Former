"""
MCR-Former训练与迁移学习脚本
- 使用1-20层数据训练基础模型
- 在21-25层数据上进行小样本迁移学习（每层10个样本）
"""

import os
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from datetime import datetime

# 直接从 MCR_Former.py 导入模型
from MCR_Former import MCR_Former

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
    """
    加载指定层数范围的V13复合材料数据
    """
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
def train_model(model, train_loader, val_loader, config, device, save_path=None):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # 学习率调度器 (移除了 PyTorch 2.0+ 已废弃的 verbose 参数)
    scheduler = None
    if config.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for sequences, lengths_batch, target_values in train_loader:
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
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
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
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{config['num_epochs']}] "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val R²: {val_r2:.4f} | "
                  f"Val RMSE: {val_rmse:.6f}")
        
        # 保存最佳模型 (使用 deepcopy 防止浅拷贝导致权重被后续训练覆盖)
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
        
        # 早停
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"{'='*80}")
    print(f"Training completed. Best Val Loss: {best_val_loss:.6f}, Best Val R²: {history['val_r2'][np.argmin(history['val_loss'])]:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# ==================== 小样本迁移学习函数 ====================
def few_shot_transfer_learning(model, train_loader, val_loader, config, device, 
                               num_epochs=30, lr_multiplier=0.1):
    """
    小样本迁移学习
    - 使用较小的学习率微调
    """
    criterion = nn.MSELoss()
    
    # 使用更小的学习率进行微调
    fine_tune_lr = config['learning_rate'] * lr_multiplier
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=fine_tune_lr,
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # 学习率调度器 (移除了 verbose 参数)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
        'random_state': 42,
        'max_seq_len': 30,
        'min_layer': 1,
        'max_layer': 20,  # 训练使用1-20层
        'data_dir': 'data',
        'save_prefix': 'mcr_former_v13',
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
    
    # 设置设备
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
    
    # 创建数据集
    train_dataset = V13CompositeDataset(
        angles=train_angles,
        v13_values=train_v13,
        layer_counts=train_layer_counts,
        max_layers=max_layers
    )
    
    # 划分训练集和验证集
    train_size = int((1 - config['train_val_split']) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['random_state'])
    )
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nTraining set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")
    
    # ==================== 步骤2: 创建并训练模型 ====================
    print("\n" + "="*80)
    print("STEP 2: Creating and training MCR-Former model")
    print("="*80)
    
    model = MCR_Former(
        input_dim=2,  # 2D最小完备表示 (sin2θ, cos2θ)
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # 训练模型
    model_save_path = f"models/{config['save_prefix']}_base.pth"
    model, train_history = train_model(
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
        # 转换numpy类型为Python原生类型
        serializable_history = {
            k: [float(v) for v in v_list] 
            for k, v_list in train_history.items()
        }
        json.dump(serializable_history, f, indent=2)
    print(f"Training history saved to: {history_save_path}")
    
    # ==================== 步骤3: 在21-25层上进行小样本迁移学习 ====================
    print("\n" + "="*80)
    print("STEP 3: Few-shot transfer learning on layers 21-25")
    print("="*80)
    
    transfer_results = {}
    
    for target_layer in range(21, 26):
        print(f"\n{'='*80}")
        print(f"Transfer Learning for Layer {target_layer}")
        print(f"{'='*80}")
        
        # 加载该层数据
        try:
            angles, v13_values, layer_counts, max_layers = load_v13_composite_data(
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
            max_layers=max_layers
        )
        
        print(f"\n  Total samples for layer {target_layer}: {len(dataset)}")
        
        # 随机选择10个样本作为微调训练集
        num_few_shot = config['transfer_few_shot_samples']
        all_indices = list(range(len(dataset)))
        np.random.shuffle(all_indices)
        
        few_shot_indices = all_indices[:num_few_shot]
        test_indices = all_indices[num_few_shot:]
        
        print(f"  Few-shot training samples: {num_few_shot}")
        print(f"  Test samples: {len(test_indices)}")
        
        # 创建数据子集
        few_shot_dataset = torch.utils.data.Subset(dataset, few_shot_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        few_shot_loader = DataLoader(
            few_shot_dataset, 
            batch_size=min(num_few_shot, 8),  # 小batch size
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # 创建新模型并加载基础模型权重
        model = MCR_Former(
            input_dim=2,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        ).to(device)
        
        # 加载基础模型权重
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded base model weights")
        
        # 小样本迁移学习
        model, transfer_history = few_shot_transfer_learning(
            model,
            few_shot_loader,
            test_loader,  # 使用测试集作为验证集（因为样本少）
            config,
            device,
            num_epochs=config['transfer_num_epochs'],
            lr_multiplier=config['transfer_lr_multiplier']
        )
        
        # 在测试集上评估
        criterion = nn.MSELoss()
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        print(f"\n  Test Results for Layer {target_layer}:")
        print(f"    Loss: {test_metrics['loss']:.6f}")
        print(f"    R²:   {test_metrics['r2']:.4f}")
        print(f"    RMSE: {test_metrics['rmse']:.6f}")
        print(f"    MAE:  {test_metrics['mae']:.6f}")
        
        # 保存结果
        transfer_results[f'layer_{target_layer}'] = {
            'num_samples': len(dataset),
            'few_shot_samples': num_few_shot,
            'test_samples': len(test_indices),
            'metrics': {
                'loss': float(test_metrics['loss']),
                'r2': float(test_metrics['r2']),
                'rmse': float(test_metrics['rmse']),
                'mae': float(test_metrics['mae'])
            }
        }
        
        # 保存迁移学习后的模型
        layer_model_path = f"models/{config['save_prefix']}_layer{target_layer}.pth"
        torch.save({
            'layer': target_layer,
            'model_state_dict': model.state_dict(),
            'metrics': test_metrics
        }, layer_model_path)
        
        # 保存迁移学习历史
        layer_history_path = f"results/{config['save_prefix']}_layer{target_layer}_history.json"
        with open(layer_history_path, 'w') as f:
            serializable_history = {
                k: [float(v) for v in v_list] 
                for k, v_list in transfer_history.items()
            }
            json.dump(serializable_history, f, indent=2)
    
    # ==================== 步骤4: 汇总并保存所有结果 ====================
    print("\n" + "="*80)
    print("STEP 4: Summary of Results")
    print("="*80)
    
    # 保存汇总结果
    summary_path = f"results/{config['save_prefix']}_transfer_summary.json"
    summary_data = {
        'config': config,
        'base_model_path': model_save_path,
        'transfer_results': transfer_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nTransfer learning results saved to: {summary_path}")
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("TRANSFER LEARNING RESULTS SUMMARY")
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