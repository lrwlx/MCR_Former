"""
深度小样本迁移学习分析脚本 (Linear Probing 版本)
核心策略：冻结预训练 Transformer 主干，仅微调最后的回归头 (Linear Probing)，
以防止极小样本下的灾难性遗忘。
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
from sklearn.metrics import r2_score
import json

# ==================== 屏蔽 PyTorch Transformer 警告 ====================
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

# 导入模型架构
from MCR_Former_concat import MCR_Former_Concat_2D

# ==================== 随机种子设置 ====================
def set_seed(seed):
    """设置随机种子以保证实验可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 数据集类 ====================
class V13CompositeDataset(Dataset):
    def __init__(self, angles, v13_values, layer_counts, max_layers=30):
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
                # 截断防止浮点误差溢出
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
    all_angles = []
    all_v13_values = []
    all_layer_counts = []
    max_layers_found = 0
    
    for layer_count in range(min_layer, max_layer + 1):
        filename = os.path.join(data_dir, f'layer_{layer_count}.csv')
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            max_layers_found = max(max_layers_found, layer_count)
        else:
            print(f"Warning: File {filename} not found, skipping layer {layer_count}")
            continue
            
        angles = df.iloc[:, :-1].values
        v13_values = df.iloc[:, -1].values
        num_samples = len(angles)
        
        padded_angles = np.zeros((num_samples, max_layers_found))
        padded_angles[:, :layer_count] = angles[:, :layer_count]
        
        layer_counts = np.full(num_samples, layer_count)
        
        all_angles.append(padded_angles)
        all_v13_values.append(v13_values)
        all_layer_counts.append(layer_counts)

    if len(all_angles) == 0:
        raise ValueError(f"No data found for layers {min_layer} to {max_layer}")

    angles = np.vstack(all_angles)
    v13_values = np.concatenate(all_v13_values)
    layer_counts = np.concatenate(all_layer_counts)

    return angles, v13_values, layer_counts, max_layers_found

# ==================== 评估函数 ====================
def evaluate_r2(model, dataloader, device):
    """评估模型在数据集上的 R² 分数"""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for sequences, lengths_batch, target_values in dataloader:
            sequences = sequences.to(device)
            lengths_batch = lengths_batch.to(device)
            target_values = target_values.to(device)
            
            outputs = model(sequences, lengths_batch)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target_values.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    if np.var(targets) < 1e-8:
        return 0.0
        
    return r2_score(targets, predictions)

# ==================== 微调函数 (Linear Probing 核心修改) ====================
def finetune_model(model, train_loader, val_loader, device, lr=0.001, num_epochs=100, patience=15, freeze_backbone=True):
    """
    小样本微调 (支持 Linear Probing)
    """
    # ==================== 核心：冻结主干网络 ====================
    trainable_params = []
    if freeze_backbone:
        # ⚠️ 请根据您的 MCR_Former_Concat_2D 模型实际结构修改这里的列表
        # 通常回归头会包含 'fc', 'linear', 'head', 'regressor', 'output' 等字眼
        keywords_to_train = ['fc', 'linear', 'head', 'regressor', 'output']
        
        for name, param in model.named_parameters():
            if any(keyword in name.lower() for keyword in keywords_to_train):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
                
        # 安全检查：防止关键字不匹配导致全模型被冻结
        if len(trainable_params) == 0:
            print("⚠️ Warning: No trainable parameters found based on keywords. Unfreezing all parameters.")
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = list(model.parameters())
    else:
        trainable_params = list(model.parameters())
    # ==========================================================

    criterion = nn.MSELoss()
    # Linear Probing 时学习率可以适当放大 (如 1e-3)，因为只训练很少的参数
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0

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
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths_batch, target_values in val_loader:
                sequences = sequences.to(device)
                lengths_batch = lengths_batch.to(device)
                target_values = target_values.to(device)
                
                outputs = model(sequences, lengths_batch)
                loss = criterion(outputs, target_values)
                val_loss += loss.item() * sequences.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model

# ==================== 主程序 ====================
def main():
    # 配置参数
    config = {
        'd_model': 64,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'max_seq_len': 30,
        'base_model_path': 'models/mcr_former_concat_v13_base.pth',
        'data_dir': 'data',
        'target_layers': range(21, 31),  # 21 到 30 层
        'sample_sizes': [0, 1, 2, 3, 5, 7, 10, 100],
        'repeat_times': 5,  # 小样本重复实验次数
        'finetune_lr': 0.001,       # Linear Probing 学习率调大至 1e-3
        'finetune_epochs': 100,
        'finetune_patience': 15,    # 线性层收敛快，patience 可适当减小
        'batch_size_train': 2,      
        'batch_size_eval': 32,
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建结果目录
    os.makedirs('results_few_shot_linear_probing', exist_ok=True)

    # 加载基础模型
    print(f"\nLoading base model from {config['base_model_path']}...")
    base_model = MCR_Former_Concat_2D(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        norm_first=True
    ).to(device)

    checkpoint = torch.load(config['base_model_path'], map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    print("Base model loaded successfully.")

    # 结果存储字典
    all_results = {}

    print("\n" + "="*80)
    print("Starting Deep Few-Shot Transfer Learning Analysis (Linear Probing)")
    print("="*80)

    for target_layer in config['target_layers']:
        print(f"\n{'='*80}")
        print(f"Analyzing Layer {target_layer}")
        print(f"{'='*80}")
        
        # 加载该层数据
        try:
            angles, v13_values, layer_counts, max_layers = load_v13_composite_data(
                min_layer=target_layer,
                max_layer=target_layer,
                data_dir=config['data_dir']
            )
        except ValueError as e:
            print(f"Skipping layer {target_layer}: {e}")
            continue
            
        total_samples = len(angles)
        print(f"Total samples available: {total_samples}")
        
        all_results[target_layer] = {}
        
        for n_samples in config['sample_sizes']:
            print(f"\n  --- Sample Size: {n_samples} ---")
            
            if n_samples == 0:
                # Zero-shot
                dataset = V13CompositeDataset(angles, v13_values, layer_counts, max_layers)
                test_loader = DataLoader(dataset, batch_size=config['batch_size_eval'], shuffle=False)
                
                r2 = evaluate_r2(base_model, test_loader, device)
                all_results[target_layer][n_samples] = [r2]
                print(f"    Zero-shot R²: {r2:.4f}")
                
            elif n_samples == 100:
                # 100-shot
                set_seed(42)
                indices = np.random.permutation(total_samples)
                train_indices = indices[:n_samples]
                test_indices = indices[n_samples:]
                
                train_dataset = torch.utils.data.Subset(
                    V13CompositeDataset(angles, v13_values, layer_counts, max_layers), train_indices)
                test_dataset = torch.utils.data.Subset(
                    V13CompositeDataset(angles, v13_values, layer_counts, max_layers), test_indices)
                
                train_loader = DataLoader(train_dataset, batch_size=min(config['batch_size_train'], n_samples), shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], shuffle=False)
                
                model = MCR_Former_Concat_2D(
                    d_model=config['d_model'], nhead=config['nhead'], 
                    num_layers=config['num_layers'], dim_feedforward=config['dim_feedforward'],
                    dropout=config['dropout'], max_seq_len=config['max_seq_len'], norm_first=True
                ).to(device)
                model.load_state_dict(base_model.state_dict())
                
                model = finetune_model(
                    model, train_loader, test_loader, device,
                    lr=config['finetune_lr'], num_epochs=config['finetune_epochs'], 
                    patience=config['finetune_patience'], freeze_backbone=True
                )
                
                r2 = evaluate_r2(model, test_loader, device)
                all_results[target_layer][n_samples] = [r2]
                print(f"    100-shot R²: {r2:.4f}")
                
            else:
                # 1, 2, 3, 5, 7, 10: 重复5次
                r2_scores = []
                for repeat_idx in range(config['repeat_times']):
                    set_seed(repeat_idx * 100 + target_layer)
                    
                    indices = np.random.permutation(total_samples)
                    train_indices = indices[:n_samples]
                    test_indices = indices[n_samples:]
                    
                    train_dataset = torch.utils.data.Subset(
                        V13CompositeDataset(angles, v13_values, layer_counts, max_layers), train_indices)
                    test_dataset = torch.utils.data.Subset(
                        V13CompositeDataset(angles, v13_values, layer_counts, max_layers), test_indices)
                    
                    bs = min(2, n_samples)
                    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], shuffle=False)
                    
                    model = MCR_Former_Concat_2D(
                        d_model=config['d_model'], nhead=config['nhead'], 
                        num_layers=config['num_layers'], dim_feedforward=config['dim_feedforward'],
                        dropout=config['dropout'], max_seq_len=config['max_seq_len'], norm_first=True
                    ).to(device)
                    model.load_state_dict(base_model.state_dict())
                    
                    model = finetune_model(
                        model, train_loader, test_loader, device,
                        lr=config['finetune_lr'], num_epochs=config['finetune_epochs'], 
                        patience=config['finetune_patience'], freeze_backbone=True
                    )
                    
                    r2 = evaluate_r2(model, test_loader, device)
                    r2_scores.append(r2)
                     
                all_results[target_layer][n_samples] = r2_scores
                mean_r2 = np.mean(r2_scores)
                std_r2 = np.std(r2_scores)
                print(f"    {n_samples}-shot R² (mean ± std): {mean_r2:.4f} ± {std_r2:.4f}")

    # ==================== 结果汇总与保存 ====================
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS (Linear Probing)")
    print("="*80)

    header = f"{'Layer': <8}"
    for n in config['sample_sizes']:
        header += f"{n: >12}"
    print(header)
    print("-" * len(header))

    summary_data = []
    for layer in config['target_layers']:
        if layer not in all_results:
            continue
        row = f"{layer: <8}"
        row_data = {'layer': layer}
        for n in config['sample_sizes']:
            scores = all_results[layer][n]
            mean_r2 = np.mean(scores)
            std_r2 = np.std(scores) if len(scores) > 1 else 0.0
            
            if n in [0, 100]:
                cell = f"{mean_r2:.4f}"
            else:
                cell = f"{mean_r2:.4f}±{std_r2:.4f}"
            row += f"{cell: >12}"
            
            row_data[f'sample_{n}_mean'] = float(mean_r2)
            row_data[f'sample_{n}_std'] = float(std_r2)
            row_data[f'sample_{n}_scores'] = [float(s) for s in scores]
            
        print(row)
        summary_data.append(row_data)

    # 保存为 CSV
    df_summary = pd.DataFrame(summary_data)
    csv_path = 'results_few_shot_linear_probing/few_shot_analysis_summary.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    # 保存为 JSON
    json_path = 'results_few_shot_linear_probing/few_shot_analysis_summary.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {json_path}")

    print("\n" + "="*80)
    print("Analysis completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()