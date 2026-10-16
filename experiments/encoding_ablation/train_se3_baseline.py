"""
SE(3)-Transformer Baseline 训练脚本
与 MCR-Former 使用相同的数据和训练配置，确保公平对比
"""

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
import matplotlib.pyplot as plt
import json
import os
import sys

# 导入 SE3-Transformer 模型
from SE3_Transformer_Baseline import SE3TransformerBaseline

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class V13Dataset(Dataset):
    """
    与 MCR-Former 相同的数据集类
    但输入是原始角度（不应用 MCR 编码）
    """
    def __init__(self, angles, v13_values, lengths, num_layers_angle):
        num_samples, seq_len = angles.shape
        assert seq_len >= num_layers_angle, f"数据中的角度列数 ({seq_len}) 小于指定的层数 ({num_layers_angle})"
        
        angles = angles[:, :num_layers_angle]
        self.num_layers_angle = num_layers_angle
        
        # SE3-Transformer 使用原始角度（弧度）
        self.angle_features = np.radians(angles)  # 转换为弧度
        self.original_angles = angles  # 保存原始角度
        
        self.v13_values = v13_values
        self.lengths = lengths
    
    def __len__(self):
        return len(self.v13_values)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.angle_features[idx], dtype=torch.float32).unsqueeze(-1),  # (seq_len, 1)
            torch.tensor(self.lengths[idx], dtype=torch.int64),
            torch.tensor(self.v13_values[idx], dtype=torch.float32),
            torch.tensor(self.original_angles[idx], dtype=torch.float32)
        )


def load_v13_data(file_path, num_layers_angle, max_seq_len=20):
    """加载数据（与 MCR-Former 相同）"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    available_angle_cols = [col for col in df.columns if col.startswith('angle_')]
    available_layers = len(available_angle_cols)
    
    if num_layers_angle > available_layers:
        raise ValueError(f"请求的层数 ({num_layers_angle}) 超过数据文件中的可用层数 ({available_layers})")
    
    print(f"使用前 {num_layers_angle} 层角度数据")
    
    angle_columns = [f'angle_{i+1}' for i in range(num_layers_angle)]
    angles = df[angle_columns].values
    v13_values = df['v13'].values
    
    print(f"Angle range: {angles.min():.2f}° to {angles.max():.2f}°")
    print(f"v13 range: {v13_values.min():.6f} to {v13_values.max():.6f}")
    
    lengths = np.full(len(angles), num_layers_angle)
    
    return angles, v13_values, lengths


def train_and_evaluate_se3(config, use_augmentation=True):
    """SE3-Transformer 训练和评估流程"""
    print(f"\n{'='*60}")
    print(f"SE3-Transformer Baseline 训练开始 - 数据增强：{'启用' if use_augmentation else '禁用'}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    if not os.path.exists(config['data_file']):
        print(f"错误：数据文件 {config['data_file']} 未找到!")
        return None
    
    angles, v13_values, lengths = load_v13_data(
        config['data_file'], 
        config['num_layers_angle'],
        config['max_seq_len']
    )
    
    # 2. v13 值归一化
    v13_scaler = StandardScaler()
    v13_scaled = v13_scaler.fit_transform(v13_values.reshape(-1, 1)).flatten()
    
    # 3. 划分数据集
    train_idx, test_idx = train_test_split(
        np.arange(len(angles)), 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    # 数据增强
    if use_augmentation:
        print(f"\n应用数据增强：扩展训练集，包含反向序列...")
        
        train_angles_orig = angles[train_idx]
        train_v13_orig = v13_scaled[train_idx]
        train_lengths_orig = lengths[train_idx]
        
        train_angles_reversed = np.flip(train_angles_orig, axis=1)
        train_v13_reversed = train_v13_orig.copy()
        train_lengths_reversed = train_lengths_orig.copy()
        
        train_angles = np.vstack([train_angles_orig, train_angles_reversed])
        train_v13 = np.hstack([train_v13_orig, train_v13_reversed])
        train_lengths = np.hstack([train_lengths_orig, train_lengths_reversed])
        
        print(f"原始训练样本：{len(train_angles_orig)}")
        print(f"反向训练样本：{len(train_angles_reversed)}")
        print(f"扩展后训练样本总数：{len(train_angles)}")
    else:
        train_angles = angles[train_idx]
        train_v13 = v13_scaled[train_idx]
        train_lengths = lengths[train_idx]
        print(f"\n未使用数据增强")
        print(f"训练样本：{len(train_angles)}")
    
    test_angles = angles[test_idx]
    test_v13 = v13_scaled[test_idx]
    test_lengths = lengths[test_idx]
    
    print(f"测试样本：{len(test_angles)}")
    
    # 4. 创建数据集和加载器
    train_dataset = V13Dataset(train_angles, train_v13, train_lengths, config['num_layers_angle'])
    test_dataset = V13Dataset(test_angles, test_v13, test_lengths, config['num_layers_angle'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"训练批次数量：{len(train_loader)}")
    print(f"测试批次数量：{len(test_loader)}")
    
    # 5. 初始化 SE3-Transformer 模型
    model = SE3TransformerBaseline(
        input_dim=1,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"总参数：{total_params:,}")
    print(f"可训练参数：{trainable_params:,}")
    
    # 6. 训练模型
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\n开始训练 SE3-Transformer...")
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for sequences, lengths_batch, v13_targets, _ in train_loader:
            sequences, v13_targets = sequences.to(device), v13_targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths_batch)
            loss = criterion(outputs, v13_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths_batch, v13_targets, _ in test_loader:
                sequences, v13_targets = sequences.to(device), v13_targets.to(device)
                lengths_batch = lengths_batch.to(device)
                
                outputs = model(sequences, lengths_batch)
                loss = criterion(outputs, v13_targets)
                val_loss += loss.item() * sequences.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_suffix = "_se3_with_aug" if use_augmentation else "_se3_no_aug"
            model_filename = f"{config['model_save_base']}{model_suffix}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': v13_scaler.mean_[0],
                'scaler_scale': v13_scaler.scale_[0],
                'config': config,
                'num_layers_angle': config['num_layers_angle'],
                'epoch': epoch,
                'val_loss': val_loss
            }, model_filename)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {current_lr:.6f}')
    
    print(f"\n训练完成！最佳验证损失：{best_val_loss:.6f}")
    
    # 7. 评估最佳模型
    print("\n加载最佳模型进行测试集评估...")
    model_suffix = "_se3_with_aug" if use_augmentation else "_se3_no_aug"
    model_filename = f"{config['model_save_base']}{model_suffix}.pth"
    
    checkpoint = torch.load(model_filename, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 测试集预测
    all_preds, all_targets = [], []
    all_original_preds, all_reversed_preds = [], []
    all_angles = []
    
    with torch.no_grad():
        for sequences, lengths_batch, v13_targets, original_angles in test_loader:
            sequences, v13_targets = sequences.to(device), v13_targets.to(device)
            lengths_batch = lengths_batch.to(device)
            
            # 原始序列预测
            original_pred = model(sequences, lengths_batch)
            all_original_preds.extend(original_pred.cpu().numpy())
            
            # 反转序列预测
            reversed_sequences = torch.flip(sequences, dims=[1])
            reversed_pred = model(reversed_sequences, lengths_batch)
            all_reversed_preds.extend(reversed_pred.cpu().numpy())
            
            all_preds.extend(original_pred.cpu().numpy())
            all_targets.extend(v13_targets.cpu().numpy())
            all_angles.extend(original_angles.cpu().numpy())
    
    # 反归一化
    all_preds_original = v13_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    all_targets_original = v13_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
    all_original_preds_original = v13_scaler.inverse_transform(np.array(all_original_preds).reshape(-1, 1)).flatten()
    all_reversed_preds_original = v13_scaler.inverse_transform(np.array(all_reversed_preds).reshape(-1, 1)).flatten()
    
    # 计算评估指标
    r2 = r2_score(all_targets_original, all_preds_original)
    mse = mean_squared_error(all_targets_original, all_preds_original)
    rmse = np.sqrt(mse)
    
    augmentation_status = "（使用数据增强）" if use_augmentation else "（未使用数据增强）"
    print(f"\n测试集评估结果{augmentation_status}:")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    # 8. 保存训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'SE3-Transformer - Training History ({augmentation_status})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    fig_suffix = "_se3_with_aug" if use_augmentation else "_se3_no_aug"
    training_history_filename = f"{config['model_save_base']}_training_history{fig_suffix}.png"
    plt.savefig(training_history_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. 保存 R² 散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(all_targets_original, all_preds_original, alpha=0.6, color='green', s=15)
    plt.plot([min(all_targets_original), max(all_targets_original)], 
             [min(all_targets_original), max(all_targets_original)], 'r--', linewidth=2) 
    plt.xlabel('Actual v13', fontsize=12)
    plt.ylabel('Predicted v13', fontsize=12)
    plt.title(f'SE3-Transformer - Prediction vs Actual ({augmentation_status}, R² = {r2:.4f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    prediction_plot_filename = f"{config['model_save_base']}_prediction_vs_actual{fig_suffix}.png"
    plt.savefig(prediction_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. 对称性验证
    symmetry_results = pd.DataFrame({
        'sample_index': range(len(test_dataset)),
        'ground_truth': all_targets_original.astype(float),
        'forward_prediction': all_original_preds_original.astype(float),
        'reversed_prediction': all_reversed_preds_original.astype(float),
        'forward_error': np.abs(all_original_preds_original - all_targets_original).astype(float),
        'reversed_error': np.abs(all_reversed_preds_original - all_targets_original).astype(float),
        'symmetry_diff': np.abs(all_original_preds_original - all_reversed_preds_original).astype(float)
    })
    
    for i in range(config['num_layers_angle']):
        symmetry_results[f'angle_{i+1}'] = [float(angles[i]) for angles in all_angles]
    
    result_suffix = "_se3_with_aug" if use_augmentation else "_se3_no_aug"
    symmetry_results_filename = f"{config['model_save_base']}_symmetry_validation_results{result_suffix}.xlsx"
    symmetry_results.to_excel(symmetry_results_filename, index=False)
    print(f"\n对称性验证结果已保存到 '{symmetry_results_filename}'")
    
    # 11. 保存评估结果
    results = {
        'test_r2': float(r2),
        'test_mse': float(mse),
        'test_rmse': float(rmse),
        'symmetry_stats': {
            'mean_diff': float(symmetry_results['symmetry_diff'].mean()),
            'max_diff': float(symmetry_results['symmetry_diff'].max()),
            'min_diff': float(symmetry_results['symmetry_diff'].min()),
            'std_diff': float(symmetry_results['symmetry_diff'].std()),
            'median_diff': float(symmetry_results['symmetry_diff'].median())
        },
        'config': config,
        'augmentation': use_augmentation,
        'model_type': 'SE3-Transformer-Baseline'
    }
    
    results_filename = f"{config['model_save_base']}_results{result_suffix}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n评估结果已保存为 '{results_filename}'")
    
    # 12. 打印对称性验证结果
    print(f"\n对称性验证统计:")
    print(f"  平均预测差异：{symmetry_results['symmetry_diff'].mean():.8f}")
    print(f"  最大预测差异：{symmetry_results['symmetry_diff'].max():.8f}")
    print(f"  标准偏差：{symmetry_results['symmetry_diff'].std():.8f}")
    
    print(f"\n✓ SE3-Transformer 训练和验证完成{augmentation_status}")
    
    return results


if __name__ == "__main__":
    # 配置参数（与 MCR-Former 保持一致）
    config = {
        'batch_size': 32,
        'd_model': 64,
        'nhead': 2,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.09,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'test_size': 0.2,
        'random_state': 42,
        'max_seq_len': 20,
        'num_layers_angle': 16,
        'data_file': 'composite_v13_dataset.csv',
        'model_save_base': 'se3_transformer_best_model'
    }
    
    print("SE3-Transformer Baseline: 复合材料 V13 预测模型")
    print("="*50)
    
    # 运行训练
    results_with_aug = train_and_evaluate_se3(config, use_augmentation=True)
    
    print("\n" + "="*60)
    print("训练完成！所有结果已保存到相应文件中。")
    print("="*60)