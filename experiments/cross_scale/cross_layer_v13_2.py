"""
MCR-Former 样本量消融实验脚本
计算不同样本量 (25, 50, 100, 200, 300 samples/layer) 下，
1-20层复合材料模型的总体 R^2 和 RMSE，并保存数据。
"""
import os
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

# 假设 MCR_Former_concat.py 在同一目录下
from MCR_Former_concat import MCR_Former_Concat_2D

# ==================== 数据集类 ====================
class V13CompositeDataset(Dataset):
    def __init__(self, features, v13_values, layer_counts, max_layers=20):
        self.features = features
        self.v13_values = v13_values
        self.layer_counts = layer_counts
        self.max_layers = max_layers
        
    def __len__(self):
        return len(self.v13_values)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.layer_counts[idx], dtype=torch.int64),
            torch.tensor(self.v13_values[idx], dtype=torch.float32)
        )

# ==================== 工具函数 ====================
def reverse_features(features, layer_counts):
    reversed_feats = np.zeros_like(features)
    for i in range(len(features)):
        l = layer_counts[i]
        reversed_feats[i, :l, :] = features[i, :l, :][::-1, :]
    return reversed_feats

def build_mcr_features(angles, layer_count, max_layers):
    num_samples = len(angles)
    features = np.zeros((num_samples, max_layers, 2))
    for i in range(num_samples):
        for j in range(layer_count):
            theta = math.radians(angles[i, j])
            features[i, j, 0] = math.sin(2 * theta)
            features[i, j, 1] = math.cos(2 * theta)
    return features

# ==================== 数据准备函数 ====================
def prepare_data_for_sample_size(min_layer, max_layer, samples_per_layer, test_ratio=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    max_layers = max_layer
    
    all_train_feats, all_train_v13, all_train_layers = [], [], []
    all_test_feats, all_test_v13, all_test_layers = [], [], []
    
    for layer_count in range(min_layer, max_layer + 1):
        filename = f'data/layer_{layer_count}.csv'
        if not os.path.exists(filename):
            continue
            
        df = pd.read_csv(filename)
        angles = df.iloc[:, :-1].values
        v13 = df.iloc[:, -1].values
        
        total_available = len(angles)
        n_samples = min(total_available, samples_per_layer)
        
        # 固定随机排列，保证数据嵌套性
        indices = rng.permutation(total_available)[:n_samples]
        angles = angles[indices]
        v13 = v13[indices]
        
        # 80% 训练，20% 测试
        n_test = int(n_samples * test_ratio)
        split_idx = n_samples - n_test
        
        train_angles, train_v13 = angles[:split_idx], v13[:split_idx]
        test_angles, test_v13 = angles[split_idx:], v13[split_idx:]
        
        train_feats = build_mcr_features(train_angles, layer_count, max_layers)
        test_feats = build_mcr_features(test_angles, layer_count, max_layers)
        
        all_train_feats.append(train_feats)
        all_train_v13.append(train_v13)
        all_train_layers.append(np.full(len(train_v13), layer_count))
        
        all_test_feats.append(test_feats)
        all_test_v13.append(test_v13)
        all_test_layers.append(np.full(len(test_v13), layer_count))
        
    train_feats = np.vstack(all_train_feats)
    train_v13 = np.concatenate(all_train_v13)
    train_layers = np.concatenate(all_train_layers)
    
    test_feats = np.vstack(all_test_feats)
    test_v13 = np.concatenate(all_test_v13)
    test_layers = np.concatenate(all_test_layers)
    
    return (train_feats, train_v13, train_layers), (test_feats, test_v13, test_layers), max_layers

# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        for feats, lengths, targets in train_loader:
            feats, lengths, targets = feats.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(feats, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, lengths, targets in val_loader:
                feats, lengths, targets = feats.to(device), lengths.to(device), targets.to(device)
                outputs = model(feats, lengths)
                val_loss += criterion(outputs, targets).item() * feats.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.get('early_stop_patience', 15):
            break
            
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

# ==================== 主流程 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
        'batch_size': 64, 'd_model': 64, 'nhead': 8, 'num_layers': 4,
        'dim_feedforward': 256, 'dropout': 0.1, 'learning_rate': 0.001,
        'num_epochs': 100, 'weight_decay': 1e-4, 'early_stop_patience': 15
    }
    
    # 用户指定的样本量
    samples_list = [25, 50, 100, 200, 300]
    results = []
    
    for spl in samples_list:
        print(f"\n{'='*60}")
        print(f"Training with {spl} samples/layer (1-20 layers)")
        print(f"{'='*60}")
        
        # 1. 准备数据
        (train_feats, train_v13, train_layers), (test_feats, test_v13, test_layers), max_layers = \
            prepare_data_for_sample_size(1, 20, spl, test_ratio=0.2, random_state=42)
            
        # 2. 划分 Train/Val (从 80% 的训练池中再分 10% 做验证)
        train_pool_idx, val_idx = train_test_split(
            np.arange(len(train_v13)), test_size=0.1, random_state=42
        )
        
        # 3. 数据增强 (反转)
        train_pool_feats = train_feats[train_pool_idx]
        train_pool_v13 = train_v13[train_pool_idx]
        train_pool_layers = train_layers[train_pool_idx]
        
        train_pool_feats_rev = reverse_features(train_pool_feats, train_pool_layers)
        aug_feats = np.vstack([train_pool_feats, train_pool_feats_rev])
        aug_v13 = np.hstack([train_pool_v13, train_pool_v13])
        aug_layers = np.hstack([train_pool_layers, train_pool_layers])
        
        val_feats = train_feats[val_idx]
        val_v13 = train_v13[val_idx]
        val_layers = train_layers[val_idx]
        
        # 4. 创建 Loaders
        train_loader = DataLoader(V13CompositeDataset(aug_feats, aug_v13, aug_layers, max_layers), batch_size=64, shuffle=True)
        val_loader = DataLoader(V13CompositeDataset(val_feats, val_v13, val_layers, max_layers), batch_size=64, shuffle=False)
        test_loader = DataLoader(V13CompositeDataset(test_feats, test_v13, test_layers, max_layers), batch_size=128, shuffle=False)
        
        # 5. 训练模型
        model = MCR_Former_Concat_2D(
            d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'], dropout=config['dropout'], max_seq_len=20, norm_first=True
        ).to(device)
        
        model = train_model(model, train_loader, val_loader, config, device)
        
        # 6. 在独立测试集上计算总体 R^2 和 RMSE
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for feats, lengths, targets in test_loader:
                feats, lengths = feats.to(device), lengths.to(device)
                preds = model(feats, lengths)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        overall_r2 = r2_score(all_targets, all_preds)
        overall_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        print(f"  Samples/Layer: {spl} | Overall R^2: {overall_r2:.4f} | Overall RMSE: {overall_rmse:.6f}")
        results.append({'Samples_Per_Layer': spl, 'R2': overall_r2, 'RMSE': overall_rmse})
        
    # 保存结果
    df_results = pd.DataFrame(results)
    csv_path = 'sample_size_ablation_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

if __name__ == "__main__":
    main()