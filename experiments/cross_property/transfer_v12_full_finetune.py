"""
MCR-Former从v13到v12的迁移学习脚本
全量微调策略，测试不同数据比例下的迁移效果
"""
import os
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from tqdm import tqdm
from MCR_Former_concat import MCR_Former_Concat_2D

# ==================== 数据集类 ====================
class V12CompositeDataset(Dataset):
    def __init__(self, angles, v12_values, layer_counts, max_layers=30):
        """
        angles: [num_samples, max_layers] - 铺层角度序列
        v12_values: [num_samples] - v12泊松比值
        layer_counts: [num_samples] - 每个样本的实际层数
        """
        self.layer_counts = layer_counts
        self.max_layers = max_layers
        
        # 创建2D物理感知特征 (sin(2θ), cos(2θ))
        self.layer_features = np.zeros((len(angles), max_layers, 2))
        
        for i in range(len(angles)):
            layers = layer_counts[i]
            for j in range(layers):
                theta = math.radians(angles[i, j])
                self.layer_features[i, j, 0] = math.sin(2 * theta)
                self.layer_features[i, j, 1] = math.cos(2 * theta)
                self.layer_features[i, j, 0] = max(min(self.layer_features[i, j, 0], 1.0), -1.0)
                self.layer_features[i, j, 1] = max(min(self.layer_features[i, j, 1], 1.0), -1.0)
        
        self.v12_values = v12_values
        
    def __len__(self):
        return len(self.v12_values)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.layer_features[idx], dtype=torch.float32),
            torch.tensor(self.layer_counts[idx], dtype=torch.int64),
            torch.tensor(self.v12_values[idx], dtype=torch.float32)
        )

# ==================== 数据加载函数 ====================
def load_v12_data(data_path='data_v12/v12_dataset.csv'):
    """加载v12泊松比数据"""
    print(f"\nLoading v12 data from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples")
    
    # 分离特征和标签
    angles = df.iloc[:, :-1].values  # 铺层角度
    v12_values = df.iloc[:, -1].values  # v12值
    
    # 确定每个样本的实际层数
    layer_counts = np.zeros(len(angles), dtype=int)
    for i in range(len(angles)):
        # 找到第一个全为0的位置（假设0表示填充）
        non_zero = np.where(angles[i] != 0)[0]
        layer_counts[i] = len(non_zero) if len(non_zero) > 0 else 1
    
    max_layers = angles.shape[1]
    
    print(f"  Angle array shape: {angles.shape}")
    print(f"  Max layers: {max_layers}")
    print(f"  Layer counts range: {layer_counts.min()} to {layer_counts.max()}")
    print(f"  v12 range: {v12_values.min():.6f} to {v12_values.max():.6f}")
    print(f"  v12 mean: {v12_values.mean():.6f}, std: {v12_values.std():.6f}")
    
    return angles, v12_values, layer_counts, max_layers

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

# ==================== 全量微调函数 ====================
def full_finetune_model(model, train_loader, val_loader, config, device, 
                        save_path=None, use_layerwise_lr=False):
    """
    全量微调模型
    
    参数:
        model: 预训练模型
        use_layerwise_lr: 是否使用分层学习率
            - True: 浅层用小LR，深层用大LR
            - False: 所有层使用相同的小LR
    """
    criterion = nn.MSELoss()
    
    # 设置优化器
    if use_layerwise_lr:
        # 分层学习率策略
        # 嵌入层和位置编码：最小LR（通用特征）
        # Transformer编码器前几层：中等LR
        # Transformer编码器后几层和回归头：最大LR
        
        base_lr = config['learning_rate']
        
        param_groups = [
            # 嵌入层
            {'params': model.mcr_embedding.parameters(), 'lr': base_lr * 0.1},
            # 位置编码（通常不训练或极小LR）
            {'params': model.pos_encoder.parameters(), 'lr': base_lr * 0.01},
            # Transformer编码器前2层
            {'params': model.shared_encoder.layers[:2].parameters(), 'lr': base_lr * 0.3},
            # Transformer编码器后几层
            {'params': model.shared_encoder.layers[2:].parameters(), 'lr': base_lr * 0.7},
            # 融合层和回归头
            {'params': model.fusion_layer.parameters(), 'lr': base_lr},
            {'params': model.regression_head.parameters(), 'lr': base_lr},
        ]
        
        optimizer = optim.AdamW(param_groups, weight_decay=config.get('weight_decay', 1e-4))
    else:
        # 统一学习率（较小）
        finetune_lr = config['learning_rate'] * 0.1  # 使用原LR的1/10
        optimizer = optim.AdamW(
            model.parameters(),
            lr=finetune_lr,
            weight_decay=config.get('weight_decay', 1e-4)
        )
    
    # 学习率调度器
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
    early_stop_patience = config.get('early_stop_patience', 20)
    
    num_epochs = config.get('finetune_epochs', 100)
    
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
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
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
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
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'Val R²: {val_r2:.4f}, LRs: {[f"{lr:.6f}" for lr in current_lrs]}')
        
        # 早停
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    print(f"{'='*80}")
    print(f"Fine-tuning completed. Best Val Loss: {best_val_loss:.6f}, "
          f"Best Val R²: {history['val_r2'][np.argmin(history['val_loss'])]:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, training_time

# ==================== 从头训练函数（用于对比） ====================
def train_from_scratch(model, train_loader, val_loader, config, device, save_path=None):
    """从头训练模型（作为baseline对比）"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    grad_clip = config.get('grad_clip', 1.0)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_rmse': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = config.get('early_stop_patience', 20)
    
    num_epochs = config.get('finetune_epochs', 100)
    
    print(f"\nTraining from scratch for {num_epochs} epochs...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
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
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        
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
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'Val R²: {val_r2:.4f}, LR: {current_lr:.6f}')
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    print(f"{'='*80}")
    print(f"Training completed. Best Val Loss: {best_val_loss:.6f}, "
          f"Best Val R²: {history['val_r2'][np.argmin(history['val_loss'])]:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, training_time

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
        'learning_rate': 0.001,  # 原始训练LR
        'finetune_epochs': 100,
        'train_val_split': 0.2,
        'random_state': 42,
        'max_seq_len': 30,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'early_stop_patience': 20,
        # 预训练模型路径
        'pretrained_model_path': 'models/mcr_former_concat_v13_base.pth',
        # 数据比例实验
        'data_fractions': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],  # 测试不同数据比例
        # 是否使用分层学习率
        'use_layerwise_lr': True,
    }
    
    # 设置随机种子
    torch.manual_seed(config['random_state'])
    np.random.seed(config['random_state'])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs('models_v12', exist_ok=True)
    os.makedirs('results_v12', exist_ok=True)
    
    # ==================== 步骤1: 加载v12数据 ====================
    print("\n" + "="*80)
    print("STEP 1: Loading v12 data")
    print("="*80)
    
    angles, v12_values, layer_counts, max_layers = load_v12_data('data_v12/v12_dataset.csv')
    
    total_samples = len(angles)
    
    # 划分训练集和测试集（80%训练，20%测试）
    train_indices, test_indices = train_test_split(
        np.arange(total_samples),
        test_size=config['train_val_split'],
        random_state=config['random_state']
    )
    
    train_angles = angles[train_indices]
    train_v12 = v12_values[train_indices]
    train_layers = layer_counts[train_indices]
    
    test_angles = angles[test_indices]
    test_v12 = v12_values[test_indices]
    test_layers = layer_counts[test_indices]
    
    print(f"\nData splitting:")
    print(f"  Total samples: {total_samples}")
    print(f"  Training set: {len(train_indices)} samples")
    print(f"  Test set: {len(test_indices)} samples")
    
    # 创建测试集数据集和加载器（固定不变）
    test_dataset = V12CompositeDataset(test_angles, test_v12, test_layers, max_layers)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # ==================== 步骤2: 数据比例实验 ====================
    print("\n" + "="*80)
    print("STEP 2: Transfer learning experiments with different data fractions")
    print("="*80)
    
    experiment_results = {
        'config': config,
        'fractions': {},
        'timestamp': datetime.now().isoformat()
    }
    
    for fraction in config['data_fractions']:
        print(f"\n{'='*80}")
        print(f"Experiment with {fraction*100:.0f}% of training data")
        print(f"{'='*80}")
        
        # 按比例采样训练数据
        num_samples = int(len(train_indices) * fraction)
        rng = np.random.RandomState(config['random_state'])
        sampled_indices = rng.choice(len(train_indices), num_samples, replace=False)
        
        sampled_train_angles = train_angles[sampled_indices]
        sampled_train_v12 = train_v12[sampled_indices]
        sampled_train_layers = train_layers[sampled_indices]
        
        print(f"  Sampled {num_samples} training samples")
        
        # 数据增强：添加反向序列
        print("  Adding reversed sequences for augmentation...")
        reversed_angles = np.zeros_like(sampled_train_angles)
        for i in range(len(sampled_train_angles)):
            layer_count = sampled_train_layers[i]
            reversed_angles[i, :layer_count] = sampled_train_angles[i, :layer_count][::-1]
        
        combined_angles = np.vstack([sampled_train_angles, reversed_angles])
        combined_v12 = np.hstack([sampled_train_v12, sampled_train_v12])
        combined_layers = np.hstack([sampled_train_layers, sampled_train_layers])
        
        # 从增强后的数据中划分训练和验证
        train_sub_idx, val_sub_idx = train_test_split(
            np.arange(len(combined_angles)),
            test_size=0.2,
            random_state=config['random_state']
        )
        
        train_dataset = V12CompositeDataset(
            combined_angles[train_sub_idx],
            combined_v12[train_sub_idx],
            combined_layers[train_sub_idx],
            max_layers
        )
        val_dataset = V12CompositeDataset(
            combined_angles[val_sub_idx],
            combined_v12[val_sub_idx],
            combined_layers[val_sub_idx],
            max_layers
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        print(f"  Training samples (after augmentation): {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
        fraction_results = {
            'num_samples': num_samples,
            'augmented_samples': len(train_dataset),
            'finetune': {},
            'scratch': {}
        }
        
        # ==================== 方法1: 迁移学习（全量微调） ====================
        print(f"\n  --- Method 1: Transfer Learning (Full Fine-tuning) ---")
        
        # 创建模型并加载预训练权重
        model_ft = MCR_Former_Concat_2D(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            norm_first=True
        ).to(device)
        
        # 加载v13预训练权重
        if os.path.exists(config['pretrained_model_path']):
            checkpoint = torch.load(config['pretrained_model_path'], 
                                   map_location=device, weights_only=False)
            model_ft.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded pretrained weights from {config['pretrained_model_path']}")
            print(f"  Pretrained model val R²: {checkpoint.get('val_r2', 'N/A')}")
        else:
            print(f"  Warning: Pretrained model not found at {config['pretrained_model_path']}")
            print(f"  Training from scratch instead...")
        
        # 全量微调
        ft_save_path = f"models_v12/finetune_{fraction*100:.0f}pct.pth"
        model_ft, ft_history, ft_time = full_finetune_model(
            model_ft, train_loader, val_loader, config, device,
            save_path=ft_save_path,
            use_layerwise_lr=config['use_layerwise_lr']
        )
        
        # 在测试集上评估
        ft_test_metrics = evaluate_model(model_ft, test_loader, nn.MSELoss(), device)
        
        print(f"\n  Transfer Learning Test Results:")
        print(f"    R²:   {ft_test_metrics['r2']:.4f}")
        print(f"    RMSE: {ft_test_metrics['rmse']:.6f}")
        print(f"    MAE:  {ft_test_metrics['mae']:.6f}")
        print(f"    Time: {ft_time:.2f}s")
        
        fraction_results['finetune'] = {
            'metrics': {
                'r2': float(ft_test_metrics['r2']),
                'rmse': float(ft_test_metrics['rmse']),
                'mae': float(ft_test_metrics['mae']),
                'loss': float(ft_test_metrics['loss'])
            },
            'training_time': float(ft_time),
            'history': {k: [float(v) for v in v_list] for k, v_list in ft_history.items()}
        }
        
        # ==================== 方法2: 从头训练（Baseline） ====================
        print(f"\n  --- Method 2: Training from Scratch (Baseline) ---")
        
        # 创建新模型（随机初始化）
        model_scratch = MCR_Former_Concat_2D(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            norm_first=True
        ).to(device)
        
        # 从头训练
        scratch_save_path = f"models_v12/scratch_{fraction*100:.0f}pct.pth"
        model_scratch, scratch_history, scratch_time = train_from_scratch(
            model_scratch, train_loader, val_loader, config, device,
            save_path=scratch_save_path
        )
        
        # 在测试集上评估
        scratch_test_metrics = evaluate_model(model_scratch, test_loader, nn.MSELoss(), device)
        
        print(f"\n  Scratch Training Test Results:")
        print(f"    R²:   {scratch_test_metrics['r2']:.4f}")
        print(f"    RMSE: {scratch_test_metrics['rmse']:.6f}")
        print(f"    MAE:  {scratch_test_metrics['mae']:.6f}")
        print(f"    Time: {scratch_time:.2f}s")
        
        fraction_results['scratch'] = {
            'metrics': {
                'r2': float(scratch_test_metrics['r2']),
                'rmse': float(scratch_test_metrics['rmse']),
                'mae': float(scratch_test_metrics['mae']),
                'loss': float(scratch_test_metrics['loss'])
            },
            'training_time': float(scratch_time),
            'history': {k: [float(v) for v in v_list] for k, v_list in scratch_history.items()}
        }
        
        # 计算改进
        r2_improvement = ft_test_metrics['r2'] - scratch_test_metrics['r2']
        rmse_reduction = (scratch_test_metrics['rmse'] - ft_test_metrics['rmse']) / scratch_test_metrics['rmse'] * 100
        
        print(f"\n  Improvement from Transfer Learning:")
        print(f"    R² improvement: {r2_improvement:+.4f}")
        print(f"    RMSE reduction: {rmse_reduction:.2f}%")
        
        fraction_results['improvement'] = {
            'r2_diff': float(r2_improvement),
            'rmse_reduction_pct': float(rmse_reduction)
        }
        
        experiment_results['fractions'][f"{fraction*100:.0f}pct"] = fraction_results
    
    # ==================== 步骤3: 汇总结果 ====================
    print("\n" + "="*80)
    print("STEP 3: Summary of Results")
    print("="*80)
    
    # 保存结果
    summary_path = "results_v12/transfer_learning_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nResults saved to: {summary_path}")
    
    # 打印汇总表格
    print(f"\n{'='*80}")
    print(f"TRANSFER LEARNING vs SCRATCH TRAINING COMPARISON")
    print(f"{'='*80}")
    print(f"{'Data %':<10} {'Samples':<10} {'FT R²':<12} {'Scratch R²':<12} {'Improvement':<12} {'RMSE Red.':<12}")
    print("-"*80)
    
    for fraction_str, results in experiment_results['fractions'].items():
        fraction = float(fraction_str.replace('pct', '')) / 100
        ft_r2 = results['finetune']['metrics']['r2']
        scratch_r2 = results['scratch']['metrics']['r2']
        improvement = results['improvement']['r2_diff']
        rmse_red = results['improvement']['rmse_reduction_pct']
        
        print(f"{fraction*100:<10.0f} {results['num_samples']:<10} "
              f"{ft_r2:<12.4f} {scratch_r2:<12.4f} "
              f"{improvement:<+12.4f} {rmse_red:<12.2f}%")
    
    print("="*80)
    
    # ==================== 步骤4: 可视化 ====================
    print("\n" + "="*80)
    print("STEP 4: Generating visualizations")
    print("="*80)
    
    # 提取数据用于绘图
    fractions = []
    ft_r2s = []
    scratch_r2s = []
    ft_rmses = []
    scratch_rmses = []
    
    for fraction_str, results in experiment_results['fractions'].items():
        fraction = float(fraction_str.replace('pct', '')) / 100
        fractions.append(fraction * 100)
        ft_r2s.append(results['finetune']['metrics']['r2'])
        scratch_r2s.append(results['scratch']['metrics']['r2'])
        ft_rmses.append(results['finetune']['metrics']['rmse'])
        scratch_rmses.append(results['scratch']['metrics']['rmse'])
    
    # 绘制R²对比图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fractions, ft_r2s, 'o-', label='Transfer Learning', linewidth=2, markersize=8)
    plt.plot(fractions, scratch_r2s, 's-', label='From Scratch', linewidth=2, markersize=8)
    plt.xlabel('Training Data (%)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('R² Score vs Training Data Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(fractions)
    
    # 绘制RMSE对比图
    plt.subplot(1, 2, 2)
    plt.plot(fractions, ft_rmses, 'o-', label='Transfer Learning', linewidth=2, markersize=8)
    plt.plot(fractions, scratch_rmses, 's-', label='From Scratch', linewidth=2, markersize=8)
    plt.xlabel('Training Data (%)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE vs Training Data Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(fractions)
    
    plt.tight_layout()
    plt.savefig('results_v12/transfer_learning_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: results_v12/transfer_learning_comparison.png")
    
    # 绘制训练曲线对比（以50%数据为例）
    if '50pct' in experiment_results['fractions']:
        results_50 = experiment_results['fractions']['50pct']
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(results_50['finetune']['history']['val_r2'], 
                label='Transfer Learning', linewidth=2)
        plt.plot(results_50['scratch']['history']['val_r2'], 
                label='From Scratch', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation R²', fontsize=12)
        plt.title('Validation R² during Training (50% data)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(results_50['finetune']['history']['val_loss'], 
                label='Transfer Learning', linewidth=2)
        plt.plot(results_50['scratch']['history']['val_loss'], 
                label='From Scratch', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Validation Loss during Training (50% data)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results_v12/training_curves_50pct.png', dpi=300, bbox_inches='tight')
        print("  Saved: results_v12/training_curves_50pct.png")
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()