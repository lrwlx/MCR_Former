"""
MCR-Former Cross-Layer Training Script
针对 1-20 层复合材料，评估不同样本量 (100, 200, 300 samples/layer) 下的逐层性能。
每层数据 80% 用于训练 (含验证)，20% 用于独立测试。
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
import time
from datetime import datetime
from tqdm import tqdm

# 假设 MCR_Former_concat.py 在同一目录下
from MCR_Former_concat import MCR_Former_Concat_2D

# ==================== 数据集类 ====================
class V13CompositeDataset(Dataset):
    def __init__(self, features, v13_values, layer_counts, max_layers=20):
        """
        features: [num_samples, max_layers, 2] - 预计算的 MCR 特征 (sin2θ, cos2θ)
        v13_values: [num_samples] - V13 性能值
        layer_counts: [num_samples] - 每个样本的实际层数
        """
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
    """按实际层数反转序列特征，避免 padding 部分干扰"""
    reversed_feats = np.zeros_like(features)
    for i in range(len(features)):
        l = layer_counts[i]
        reversed_feats[i, :l, :] = features[i, :l, :][::-1, :]
    return reversed_feats

def build_mcr_features(angles, layer_count, max_layers):
    """将角度转换为 MCR 特征 (sin2θ, cos2θ) 并 padding"""
    num_samples = len(angles)
    features = np.zeros((num_samples, max_layers, 2))
    for i in range(num_samples):
        for j in range(layer_count):
            theta = math.radians(angles[i, j])
            features[i, j, 0] = math.sin(2 * theta)
            features[i, j, 1] = math.cos(2 * theta)
    return features

# ==================== 数据准备函数 ====================
def prepare_cross_layer_data(min_layer, max_layer, samples_per_layer, test_ratio=0.2, random_state=42):
    """
    按层加载数据，截取指定样本量，并按 80/20 划分训练/测试集。
    为了保证 100/200/300 样本量的嵌套关系，使用固定的随机排列。
    """
    rng = np.random.RandomState(random_state)
    max_layers = max_layer  # 固定为 20
    
    all_train_feats, all_train_v13, all_train_layers = [], [], []
    all_test_feats, all_test_v13, all_test_layers = [], [], []
    
    for layer_count in range(min_layer, max_layer + 1):
        filename = f'data/layer_{layer_count}.csv'
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping layer {layer_count}.")
            continue
            
        df = pd.read_csv(filename)
        angles = df.iloc[:, :-1].values
        v13 = df.iloc[:, -1].values
        
        total_available = len(angles)
        if total_available < samples_per_layer:
            print(f"Warning: Layer {layer_count} has only {total_available} samples (< {samples_per_layer}).")
            n_samples = total_available
        else:
            n_samples = samples_per_layer
            
        # 固定随机排列，确保不同样本量之间的数据嵌套性
        indices = rng.permutation(total_available)[:n_samples]
        angles = angles[indices]
        v13 = v13[indices]
        
        # 80% 训练池，20% 独立测试集
        n_test = int(n_samples * test_ratio)
        split_idx = n_samples - n_test
        
        train_angles, train_v13 = angles[:split_idx], v13[:split_idx]
        test_angles, test_v13 = angles[split_idx:], v13[split_idx:]
        
        # 构建特征
        train_feats = build_mcr_features(train_angles, layer_count, max_layers)
        test_feats = build_mcr_features(test_angles, layer_count, max_layers)
        
        all_train_feats.append(train_feats)
        all_train_v13.append(train_v13)
        all_train_layers.append(np.full(len(train_v13), layer_count))
        
        all_test_feats.append(test_feats)
        all_test_v13.append(test_v13)
        all_test_layers.append(np.full(len(test_v13), layer_count))
        
    # 合并所有层的数据
    train_feats = np.vstack(all_train_feats)
    train_v13 = np.concatenate(all_train_v13)
    train_layers = np.concatenate(all_train_layers)
    
    test_feats = np.vstack(all_test_feats)
    test_v13 = np.concatenate(all_test_v13)
    test_layers = np.concatenate(all_test_layers)
    
    return (train_feats, train_v13, train_layers), (test_feats, test_v13, test_layers), max_layers

# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, config, device, save_path=None):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"  Training for {config['num_epochs']} epochs...")
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        for feats, lengths, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feats, lengths, targets = feats.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(feats, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_loader.dataset)
        
        # 验证
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
            if save_path: torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            
        if patience_counter >= config.get('early_stop_patience', 15):
            print(f"    Early stopping at epoch {epoch+1}")
            break
            
    if best_model_state: model.load_state_dict(best_model_state)
    return model

# ==================== 评估函数 ====================
def evaluate_by_layer(model, test_feats, test_v13, test_layers, device, max_layers):
    model.eval()
    dataset = V13CompositeDataset(test_feats, test_v13, test_layers, max_layers)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for feats, lengths, targets in loader:
            feats, lengths = feats.to(device), lengths.to(device)
            preds = model(feats, lengths)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    results = {}
    for l in np.unique(test_layers):
        mask = test_layers == l
        if np.var(all_targets[mask]) < 1e-8:
            r2 = 0.0
        else:
            r2 = r2_score(all_targets[mask], all_preds[mask])
        rmse = np.sqrt(mean_squared_error(all_targets[mask], all_preds[mask]))
        results[int(l)] = {'r2': r2, 'rmse': rmse}
    return results

# ==================== 主流程 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
        'batch_size': 64, 'd_model': 64, 'nhead': 8, 'num_layers': 4,
        'dim_feedforward': 256, 'dropout': 0.1, 'learning_rate': 0.001,
        'num_epochs': 100, 'weight_decay': 1e-4, 'early_stop_patience': 15
    }
    
    samples_list = [100, 200, 300]
    all_results = []  # 用于保存 CSV
    
    for spl in samples_list:
        print(f"\n{'='*60}")
        print(f"Scenario: {spl} samples/layer (1-20 layers)")
        print(f"{'='*60}")
        
        # 1. 准备数据
        (train_feats, train_v13, train_layers), (test_feats, test_v13, test_layers), max_layers = \
            prepare_cross_layer_data(1, 20, spl, test_ratio=0.2, random_state=42)
            
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
        
        # 5. 训练模型
        model = MCR_Former_Concat_2D(
            d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'], dropout=config['dropout'], max_seq_len=20, norm_first=True
        ).to(device)
        
        save_path = f'models/mcr_former_cross_layer_{spl}.pth'
        os.makedirs('models', exist_ok=True)
        model = train_model(model, train_loader, val_loader, config, device, save_path)
        
        # 6. 逐层评估
        layer_results = evaluate_by_layer(model, test_feats, test_v13, test_layers, device, max_layers)
        
        # 7. 输出与保存
        print(f"\nResults for {spl} samples/layer:")
        print(f"{'Layer':<10} {'R2':<15} {'RMSE':<15}")
        for l in range(1, 21):
            if l in layer_results:
                r2 = layer_results[l]['r2']
                rmse = layer_results[l]['rmse']
                print(f"{l:<10} {r2:<15.4f} {rmse:<15.6f}")
                all_results.append({'Samples_Per_Layer': spl, 'Layer': l, 'R2': r2, 'RMSE': rmse})
                
    # 保存为 CSV 方便作图
    df_results = pd.DataFrame(all_results)
    csv_path = 'cross_layer_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

if __name__ == "__main__":
    main()