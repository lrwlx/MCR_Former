"""
MCR-Former从v13到v12的迁移学习脚本（严谨版本 - 无验证集设计）
全量微调策略，测试不同数据比例下的迁移效果

核心设计：
1. 测试集固定：始终在最初隔离的100个样本上评估
2. 无验证集：抽取的训练样本扩增后全部用于训练
3. 固定epoch数：避免早停带来的不确定性
4. 多次运行取平均：使用多个随机种子，报告均值和标准差
5. 数据比例：[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
"""
import os
import math
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from MCR_Former_concat import MCR_Former_Concat_2D

# 抑制特定的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================== 数据集类 ====================
class V12CompositeDataset(Dataset):
    def __init__(self, angles, v12_values, layer_counts, max_layers=30):
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
    
    angles = df.iloc[:, :-1].values
    v12_values = df.iloc[:, -1].values
    
    layer_counts = np.zeros(len(angles), dtype=int)
    for i in range(len(angles)):
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

# ==================== 动态batch_size计算 ====================
def get_effective_batch_size(dataset_size, max_batch_size=64):
    """根据数据集大小动态调整batch_size"""
    effective_batch_size = min(max_batch_size, max(1, dataset_size // 2))
    return effective_batch_size

# ==================== 全量微调函数（无验证集版本） ====================
def full_finetune_model_no_val(model, train_loader, config, device, 
                               use_layerwise_lr=False):
    """
    全量微调模型（无验证集版本）
    
    参数:
        model: 预训练模型
        train_loader: 训练数据加载器
        config: 配置参数
        device: 设备
        use_layerwise_lr: 是否使用分层学习率
    """
    criterion = nn.MSELoss()
    
    # 设置优化器
    if use_layerwise_lr:
        base_lr = config['learning_rate']
        
        param_groups = [
            {'params': model.mcr_embedding.parameters(), 'lr': base_lr * 0.1},
            {'params': model.pos_encoder.parameters(), 'lr': base_lr * 0.01},
            {'params': model.shared_encoder.layers[:2].parameters(), 'lr': base_lr * 0.3},
            {'params': model.shared_encoder.layers[2:].parameters(), 'lr': base_lr * 0.7},
            {'params': model.fusion_layer.parameters(), 'lr': base_lr},
            {'params': model.regression_head.parameters(), 'lr': base_lr},
        ]
        
        optimizer = optim.AdamW(param_groups, weight_decay=config.get('weight_decay', 1e-4))
    else:
        finetune_lr = config['learning_rate'] * 0.1
        optimizer = optim.AdamW(
            model.parameters(),
            lr=finetune_lr,
            weight_decay=config.get('weight_decay', 1e-4)
        )
    
    # 学习率调度器（基于epoch）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['finetune_epochs']
    )
    
    grad_clip = config.get('grad_clip', 1.0)
    num_epochs = config['finetune_epochs']
    
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
        scheduler.step()
    
    training_time = time.time() - start_time
    
    return model, training_time

# ==================== 从头训练函数（无验证集版本） ====================
def train_from_scratch_no_val(model, train_loader, config, device):
    """从头训练模型（无验证集版本）"""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['finetune_epochs']
    )
    
    grad_clip = config.get('grad_clip', 1.0)
    num_epochs = config['finetune_epochs']
    
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
        scheduler.step()
    
    training_time = time.time() - start_time
    
    return model, training_time

# ==================== 单次实验函数 ====================
def run_single_experiment(fraction, train_pool_angles, train_pool_v12, train_pool_layers,
                         test_angles, test_v12, test_layers, max_layers,
                         config, device, seed):
    """
    运行单次实验（一个数据比例，一个随机种子）
    
    流程：
    1. 从训练池中抽取 fraction*400 个样本
    2. 逆序扩增成 2*fraction*400 个样本
    3. 全部用于训练（无验证集）
    4. 在固定的100个测试样本上评估
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 步骤1：从训练池中抽取样本
    num_samples = int(len(train_pool_angles) * fraction)
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(len(train_pool_angles), num_samples, replace=False)
    
    sampled_angles = train_pool_angles[sampled_indices]
    sampled_v12 = train_pool_v12[sampled_indices]
    sampled_layers = train_pool_layers[sampled_indices]
    
    # 步骤2：逆序扩增
    reversed_angles = np.zeros_like(sampled_angles)
    for i in range(len(sampled_angles)):
        layer_count = sampled_layers[i]
        reversed_angles[i, :layer_count] = sampled_angles[i, :layer_count][::-1]
    
    combined_angles = np.vstack([sampled_angles, reversed_angles])
    combined_v12 = np.hstack([sampled_v12, sampled_v12])
    combined_layers = np.hstack([sampled_layers, sampled_layers])
    
    # 步骤3：创建训练数据集（全部用于训练，无验证集）
    train_dataset = V12CompositeDataset(
        combined_angles, combined_v12, combined_layers, max_layers
    )
    
    test_dataset = V12CompositeDataset(
        test_angles, test_v12, test_layers, max_layers
    )
    
    train_batch_size = get_effective_batch_size(len(train_dataset), config['batch_size'])
    test_batch_size = get_effective_batch_size(len(test_dataset), config['batch_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    results = {
        'num_samples': num_samples,
        'augmented_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'finetune': {},
        'scratch': {}
    }
    
    # 方法1: 迁移学习（全量微调）
    model_ft = MCR_Former_Concat_2D(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        norm_first=True
    ).to(device)
    
    if os.path.exists(config['pretrained_model_path']):
        checkpoint = torch.load(config['pretrained_model_path'], 
                               map_location=device, weights_only=False)
        model_ft.load_state_dict(checkpoint['model_state_dict'])
    
    model_ft, ft_time = full_finetune_model_no_val(
        model_ft, train_loader, config, device,
        use_layerwise_lr=config['use_layerwise_lr']
    )
    
    ft_test_metrics = evaluate_model(model_ft, test_loader, nn.MSELoss(), device)
    
    results['finetune'] = {
        'r2': float(ft_test_metrics['r2']),
        'rmse': float(ft_test_metrics['rmse']),
        'mae': float(ft_test_metrics['mae']),
        'training_time': float(ft_time)
    }
    
    # 方法2: 从头训练
    model_scratch = MCR_Former_Concat_2D(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        norm_first=True
    ).to(device)
    
    model_scratch, scratch_time = train_from_scratch_no_val(
        model_scratch, train_loader, config, device
    )
    
    scratch_test_metrics = evaluate_model(model_scratch, test_loader, nn.MSELoss(), device)
    
    results['scratch'] = {
        'r2': float(scratch_test_metrics['r2']),
        'rmse': float(scratch_test_metrics['rmse']),
        'mae': float(scratch_test_metrics['mae']),
        'training_time': float(scratch_time)
    }
    
    return results

# ==================== 主程序 ====================
def main():
    config = {
        'batch_size': 64,
        'd_model': 64,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'finetune_epochs': 100,  # 固定epoch数
        'random_state': 42,
        'max_seq_len': 30,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'pretrained_model_path': 'models/mcr_former_concat_v13_base.pth',
        'data_fractions': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        'use_layerwise_lr': True,
        'num_seeds': 5,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('models_v12', exist_ok=True)
    os.makedirs('results_v12', exist_ok=True)
    
    # ==================== 步骤1: 加载数据并固定测试集 ====================
    print("\n" + "="*80)
    print("STEP 1: Loading data and fixing test set")
    print("="*80)
    
    angles, v12_values, layer_counts, max_layers = load_v12_data('data_v12/v12_dataset.csv')
    
    total_samples = len(angles)
    
    # 固定测试集划分
    test_rng = np.random.RandomState(config['random_state'])
    test_size = int(total_samples * 0.2)
    test_indices = test_rng.choice(total_samples, test_size, replace=False)
    train_pool_indices = np.setdiff1d(np.arange(total_samples), test_indices)
    
    train_pool_angles = angles[train_pool_indices]
    train_pool_v12 = v12_values[train_pool_indices]
    train_pool_layers = layer_counts[train_pool_indices]
    
    test_angles = angles[test_indices]
    test_v12 = v12_values[test_indices]
    test_layers = layer_counts[test_indices]
    
    print(f"\nData splitting strategy:")
    print(f"  Total samples: {total_samples}")
    print(f"  Training pool: {len(train_pool_indices)} samples")
    print(f"  Independent test set: {len(test_indices)} samples (FIXED)")
    
    # ==================== 步骤2: 多次运行实验 ====================
    print("\n" + "="*80)
    print("STEP 2: Running experiments with multiple seeds")
    print("="*80)
    
    seeds = [42, 123, 456, 789, 1024][:config['num_seeds']]
    
    experiment_results = {
        'config': config,
        'seeds': seeds,
        'fractions': {},
        'timestamp': datetime.now().isoformat()
    }
    
    for fraction in config['data_fractions']:
        print(f"\n{'='*80}")
        print(f"Experiment with {fraction*100:.1f}% of training data")
        print(f"{'='*80}")
        
        fraction_results = {
            'fraction': fraction,
            'runs': [],
            'finetune': {'r2': [], 'rmse': [], 'mae': [], 'time': []},
            'scratch': {'r2': [], 'rmse': [], 'mae': [], 'time': []}
        }
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n  --- Run {seed_idx+1}/{len(seeds)} (seed={seed}) ---")
            
            run_results = run_single_experiment(
                fraction, train_pool_angles, train_pool_v12, train_pool_layers,
                test_angles, test_v12, test_layers, max_layers,
                config, device, seed
            )
            
            fraction_results['runs'].append(run_results)
            
            fraction_results['finetune']['r2'].append(run_results['finetune']['r2'])
            fraction_results['finetune']['rmse'].append(run_results['finetune']['rmse'])
            fraction_results['finetune']['mae'].append(run_results['finetune']['mae'])
            fraction_results['finetune']['time'].append(run_results['finetune']['training_time'])
            
            fraction_results['scratch']['r2'].append(run_results['scratch']['r2'])
            fraction_results['scratch']['rmse'].append(run_results['scratch']['rmse'])
            fraction_results['scratch']['mae'].append(run_results['scratch']['mae'])
            fraction_results['scratch']['time'].append(run_results['scratch']['training_time'])
            
            print(f"    FT:      R²={run_results['finetune']['r2']:.4f}, "
                  f"RMSE={run_results['finetune']['rmse']:.6f}")
            print(f"    Scratch: R²={run_results['scratch']['r2']:.4f}, "
                  f"RMSE={run_results['scratch']['rmse']:.6f}")
        
        # 计算统计信息
        fraction_results['finetune_stats'] = {
            'r2_mean': float(np.mean(fraction_results['finetune']['r2'])),
            'r2_std': float(np.std(fraction_results['finetune']['r2'])),
            'rmse_mean': float(np.mean(fraction_results['finetune']['rmse'])),
            'rmse_std': float(np.std(fraction_results['finetune']['rmse'])),
            'mae_mean': float(np.mean(fraction_results['finetune']['mae'])),
            'mae_std': float(np.std(fraction_results['finetune']['mae'])),
            'time_mean': float(np.mean(fraction_results['finetune']['time'])),
        }
        
        fraction_results['scratch_stats'] = {
            'r2_mean': float(np.mean(fraction_results['scratch']['r2'])),
            'r2_std': float(np.std(fraction_results['scratch']['r2'])),
            'rmse_mean': float(np.mean(fraction_results['scratch']['rmse'])),
            'rmse_std': float(np.std(fraction_results['scratch']['rmse'])),
            'mae_mean': float(np.mean(fraction_results['scratch']['mae'])),
            'mae_std': float(np.std(fraction_results['scratch']['mae'])),
            'time_mean': float(np.mean(fraction_results['scratch']['time'])),
        }
        
        r2_improvement = fraction_results['finetune_stats']['r2_mean'] - fraction_results['scratch_stats']['r2_mean']
        rmse_reduction = (fraction_results['scratch_stats']['rmse_mean'] - fraction_results['finetune_stats']['rmse_mean']) / fraction_results['scratch_stats']['rmse_mean'] * 100
        
        fraction_results['improvement'] = {
            'r2_diff_mean': float(r2_improvement),
            'rmse_reduction_pct': float(rmse_reduction)
        }
        
        experiment_results['fractions'][f"{fraction*100:.1f}pct"] = fraction_results
    
    # ==================== 步骤3: 汇总结果 ====================
    print("\n" + "="*80)
    print("STEP 3: Summary of Results")
    print("="*80)
    
    summary_path = "results_v12/transfer_learning_summary_no_val.json"
    with open(summary_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nResults saved to: {summary_path}")
    
    print(f"\n{'='*100}")
    print(f"TRANSFER LEARNING vs SCRATCH TRAINING COMPARISON (Mean ± Std)")
    print(f"{'='*100}")
    print(f"{'Data %':<10} {'Samples':<10} {'FT R²':<20} {'Scratch R²':<20} {'Improvement':<15} {'RMSE Red.':<12}")
    print("-"*100)
    
    for fraction_str, results in experiment_results['fractions'].items():
        fraction = float(fraction_str.replace('pct', '')) / 100
        ft_r2_mean = results['finetune_stats']['r2_mean']
        ft_r2_std = results['finetune_stats']['r2_std']
        scratch_r2_mean = results['scratch_stats']['r2_mean']
        scratch_r2_std = results['scratch_stats']['r2_std']
        improvement = results['improvement']['r2_diff_mean']
        rmse_red = results['improvement']['rmse_reduction_pct']
        
        num_samples = results['runs'][0]['num_samples']
        
        print(f"{fraction*100:<10.1f} {num_samples:<10} "
              f"{ft_r2_mean:.4f}±{ft_r2_std:.4f}  "
              f"{scratch_r2_mean:.4f}±{scratch_r2_std:.4f}  "
              f"{improvement:<+10.4f} {rmse_red:<10.2f}%")
    
    print("="*100)
    
    # ==================== 步骤4: 可视化 ====================
    print("\n" + "="*80)
    print("STEP 4: Generating visualizations")
    print("="*80)
    
    fractions = []
    ft_r2_means = []
    ft_r2_stds = []
    scratch_r2_means = []
    scratch_r2_stds = []
    ft_rmse_means = []
    ft_rmse_stds = []
    scratch_rmse_means = []
    scratch_rmse_stds = []
    
    for fraction_str, results in experiment_results['fractions'].items():
        fraction = float(fraction_str.replace('pct', '')) / 100
        fractions.append(fraction * 100)
        
        ft_r2_means.append(results['finetune_stats']['r2_mean'])
        ft_r2_stds.append(results['finetune_stats']['r2_std'])
        scratch_r2_means.append(results['scratch_stats']['r2_mean'])
        scratch_r2_stds.append(results['scratch_stats']['r2_std'])
        
        ft_rmse_means.append(results['finetune_stats']['rmse_mean'])
        ft_rmse_stds.append(results['finetune_stats']['rmse_std'])
        scratch_rmse_means.append(results['scratch_stats']['rmse_mean'])
        scratch_rmse_stds.append(results['scratch_stats']['rmse_std'])
    
    fractions = np.array(fractions)
    ft_r2_means = np.array(ft_r2_means)
    ft_r2_stds = np.array(ft_r2_stds)
    scratch_r2_means = np.array(scratch_r2_means)
    scratch_r2_stds = np.array(scratch_r2_stds)
    ft_rmse_means = np.array(ft_rmse_means)
    ft_rmse_stds = np.array(ft_rmse_stds)
    scratch_rmse_means = np.array(scratch_rmse_means)
    scratch_rmse_stds = np.array(scratch_rmse_stds)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(fractions, ft_r2_means, yerr=ft_r2_stds, 
                fmt='o-', capsize=5, capthick=2, 
                label='Transfer Learning', linewidth=2, markersize=8)
    plt.errorbar(fractions, scratch_r2_means, yerr=scratch_r2_stds, 
                fmt='s-', capsize=5, capthick=2,
                label='From Scratch', linewidth=2, markersize=8)
    plt.xlabel('Training Data (%)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('R² Score vs Training Data Size (Mean ± Std)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(fractions, rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.errorbar(fractions, ft_rmse_means, yerr=ft_rmse_stds, 
                fmt='o-', capsize=5, capthick=2,
                label='Transfer Learning', linewidth=2, markersize=8)
    plt.errorbar(fractions, scratch_rmse_means, yerr=scratch_rmse_stds, 
                fmt='s-', capsize=5, capthick=2,
                label='From Scratch', linewidth=2, markersize=8)
    plt.xlabel('Training Data (%)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE vs Training Data Size (Mean ± Std)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(fractions, rotation=45)
    
    plt.tight_layout()
    plt.savefig('results_v12/transfer_learning_comparison_no_val.png', dpi=300, bbox_inches='tight')
    print("  Saved: results_v12/transfer_learning_comparison_no_val.png")
    
    plt.figure(figsize=(10, 6))
    improvements = ft_r2_means - scratch_r2_means
    plt.bar(fractions, improvements, width=5, alpha=0.7, color='green')
    plt.xlabel('Training Data (%)', fontsize=12)
    plt.ylabel('R² Improvement', fontsize=12)
    plt.title('Transfer Learning Improvement over Scratch Training', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(fractions, rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('results_v12/improvement_bar_chart_no_val.png', dpi=300, bbox_inches='tight')
    print("  Saved: results_v12/improvement_bar_chart_no_val.png")
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()