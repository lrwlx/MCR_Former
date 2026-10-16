"""
严格复现LayupFormer核心逻辑：Tsai不变量嵌入 + 80/10/10划分
针对composite_v13_dataset.csv的V13预测任务
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
import time
import os
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现性（与LayupFormer论文一致）
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 严格遵循LayupFormer的Tsai嵌入 ====================
class LayupFormerDataset(Dataset):
    """严格复现LayupFormer的Tsai不变量嵌入（公式3）"""
    def __init__(self, angles, v13_values, num_layers=16):
        """
        angles: [num_samples, num_layers] - 原始角度数据（0-180°）
        v13_values: [num_samples] - 目标v13值
        """
        self.num_layers = num_layers
        
        # 严格复现LayupFormer公式(3): Φ(θ)=[cos θ, sin θ, cos 2θ, sin 2θ, cos 4θ, sin 4θ]
        self.features = np.zeros((len(angles), num_layers, 6))
        for i in range(len(angles)):
            for j in range(min(num_layers, angles.shape[1])):
                theta = angles[i, j]
                rad = np.radians(theta)  # 转换为弧度
                # 严格按论文顺序：cosθ, sinθ, cos2θ, sin2θ, cos4θ, sin4θ
                self.features[i, j, 0] = np.cos(rad)      # cos θ
                self.features[i, j, 1] = np.sin(rad)      # sin θ
                self.features[i, j, 2] = np.cos(2 * rad)  # cos 2θ
                self.features[i, j, 3] = np.sin(2 * rad)  # sin 2θ
                self.features[i, j, 4] = np.cos(4 * rad)  # cos 4θ
                self.features[i, j, 5] = np.sin(4 * rad)  # sin 4θ
        
        self.v13_values = v13_values
        self.input_dim = 6  # Tsai嵌入固定为6维

    def __len__(self):
        return len(self.v13_values)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.v13_values[idx], dtype=torch.float32)
        )

# ==================== 位置编码（标准实现） ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==================== Transformer模型（简化版，单目标） ====================
class LayupFormerPredictor(nn.Module):
    """简化版LayupFormer predictor（单目标v13）"""
    def __init__(self, input_dim=6, d_model=64, nhead=8, num_layers=4,
                 dim_feedforward=256, dropout=0.1, max_seq_len=20):
        super().__init__()
        # 输入投影（适配6维Tsai嵌入）
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 回归头（单目标）
        self.regressor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [batch_size, seq_len, 6]
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)
        x = self.transformer(x)  # [batch_size, seq_len, d_model]
        x = x.mean(dim=1)        # 全局平均池化
        return self.regressor(x).squeeze(-1)  # [batch_size]

# ==================== 数据加载（严格80/10/10划分） ====================
def load_and_split_data(file_path, num_layers=16, random_state=42):
    """严格遵循LayupFormer的80/10/10划分策略"""
    print(f"加载数据: {file_path}")
    
    # 检查文件存在性
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # 提取角度列
    angle_cols = [f'angle_{i+1}' for i in range(num_layers) if f'angle_{i+1}' in df.columns]
    if len(angle_cols) < num_layers:
        print(f"警告: 数据仅包含{len(angle_cols)}层，使用全部可用层")
        num_layers = len(angle_cols)
    
    angles = df[angle_cols].values
    v13_values = df['v13'].values
    
    print(f"数据形状: {angles.shape}")
    print(f"角度范围: {angles.min():.2f}° - {angles.max():.2f}°")
    print(f"v13范围: {v13_values.min():.6f} - {v13_values.max():.6f}")
    
    # 严格80/10/10划分（与LayupFormer一致）
    # 第一步：80%训练 + 20%临时
    train_idx, temp_idx = train_test_split(
        np.arange(len(angles)),
        test_size=0.20,
        random_state=random_state
    )
    
    # 第二步：临时集划分为10%验证 + 10%测试
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,  # 0.5 * 0.2 = 0.1
        random_state=random_state
    )
    
    print(f"数据划分: 训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={len(test_idx)}")
    print(f"划分比例: {len(train_idx)/len(angles):.1%} / {len(val_idx)/len(angles):.1%} / {len(test_idx)/len(angles):.1%}")
    
    return angles, v13_values, train_idx, val_idx, test_idx

# ==================== 严格遵循LayupFormer的标准化流程 ====================
def standardize_targets(train_values, val_values, test_values):
    """
    严格遵循LayupFormer 2.3.2节：仅基于训练集计算标准化参数
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()
    
    print(f"标准化参数 (仅基于训练集): μ={scaler.mean_[0]:.6f}, σ={scaler.scale_[0]:.6f}")
    return train_scaled, val_scaled, test_scaled, scaler

# ==================== 训练函数（AdamW + 早停） ====================
def train_layupformer(model, train_loader, val_loader, config, device):
    """严格遵循LayupFormer训练策略：AdamW + 早停"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度（OneCycleLR，与用户之前实验一致）
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("\n开始训练 LayupFormer (Tsai嵌入)...")
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * sequences.size(0)
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * sequences.size(0)
        val_loss /= len(val_loader.dataset)
        
        # 早停机制（与LayupFormer 2.3.2节一致）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"早停触发 @ epoch {epoch+1} (验证损失未改善 {config['patience']} 轮)")
                break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {lr:.6f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss

# ==================== 主实验函数 ====================
def evaluate_layupformer_on_v13(config):
    """严格复现LayupFormer核心逻辑的V13预测实验"""
    print("="*70)
    print("LayupFormer 核心逻辑复现实验 (Tsai不变量嵌入)")
    print("任务: 复合材料V13预测 (16层铺层)")
    print("嵌入: Φ(θ)=[cosθ, sinθ, cos2θ, sin2θ, cos4θ, sin4θ] (6维)")
    print("划分: 80%训练 / 10%验证 / 10%测试 (严格随机划分)")
    print("="*70)
    
    # 1. 加载并划分数据
    angles, v13_values, train_idx, val_idx, test_idx = load_and_split_data(
        config['data_file'],
        num_layers=config['num_layers'],
        random_state=config['random_state']
    )
    
    # 2. 严格标准化（仅基于训练集）
    v13_train, v13_val, v13_test, scaler = standardize_targets(
        v13_values[train_idx],
        v13_values[val_idx],
        v13_values[test_idx]
    )
    
    # 3. 创建数据集
    train_dataset = LayupFormerDataset(angles[train_idx], v13_train, config['num_layers'])
    val_dataset = LayupFormerDataset(angles[val_idx], v13_val, config['num_layers'])
    test_dataset = LayupFormerDataset(angles[test_idx], v13_test, config['num_layers'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"\n数据集创建完成:")
    print(f"  训练集: {len(train_dataset)} samples, input_dim={train_dataset.input_dim}")
    print(f"  验证集: {len(val_dataset)} samples")
    print(f"  测试集: {len(test_dataset)} samples")
    
    # 4. 初始化模型
    model = LayupFormerPredictor(
        input_dim=6,  # Tsai嵌入固定6维
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers_transformer'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['num_layers']
    )
    print(f"\n模型架构: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,} parameters")
    
    # 5. 训练模型
    start_time = time.time()
    model, best_val_loss = train_layupformer(model, train_loader, val_loader, config, device)
    training_time = time.time() - start_time
    print(f"\n训练完成 - 最佳验证损失: {best_val_loss:.6f}, 耗时: {training_time:.2f}秒")
    
    # 6. 测试评估（在原始空间计算指标）
    model.eval()
    test_preds_scaled = []
    test_targets_scaled = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            test_preds_scaled.extend(outputs.cpu().numpy())
            test_targets_scaled.extend(targets.cpu().numpy())
    
    # 反标准化到原始v13空间
    test_preds_original = scaler.inverse_transform(
        np.array(test_preds_scaled).reshape(-1, 1)
    ).flatten()
    test_targets_original = scaler.inverse_transform(
        np.array(test_targets_scaled).reshape(-1, 1)
    ).flatten()
    
    # 计算指标（原始空间）
    test_r2 = r2_score(test_targets_original, test_preds_original)
    test_rmse = np.sqrt(mean_squared_error(test_targets_original, test_preds_original))
    
    # 7. 输出结果
    print("\n" + "="*70)
    print("LayupFormer (Tsai嵌入) 测试集性能")
    print("="*70)
    print(f"测试集样本数: {len(test_targets_original)}")
    print(f"R² (原始空间): {test_r2:.6f}")
    print(f"RMSE (原始空间): {test_rmse:.6f}")
    print(f"v13预测范围: [{test_preds_original.min():.6f}, {test_preds_original.max():.6f}]")
    print(f"v13真实范围: [{test_targets_original.min():.6f}, {test_targets_original.max():.6f}]")
    print("="*70)
    
    # 8. 可视化预测效果
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.scatter(test_targets_original, test_preds_original, alpha=0.6, s=30, color='#1f77b4')
        plt.plot([test_targets_original.min(), test_targets_original.max()],
                [test_targets_original.min(), test_targets_original.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('True v₁₃', fontsize=14)
        plt.ylabel('Predicted v₁₃', fontsize=14)
        plt.title(f'LayupFormer (Tsai Embedding)\nR² = {test_r2:.4f}, RMSE = {test_rmse:.6f}', 
                 fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 保存结果
        os.makedirs('results/layupformer_comparison', exist_ok=True)
        plt.savefig('results/layupformer_comparison/tsai_embedding_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细结果
        results = {
            'model': 'LayupFormer_TsaiEmbedding',
            'embedding': 'Φ(θ)=[cosθ,sinθ,cos2θ,sin2θ,cos4θ,sin4θ]',
            'input_dim': 6,
            'data_split': '80/10/10',
            'test_samples': len(test_targets_original),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'training_time_seconds': training_time,
            'best_val_loss': float(best_val_loss),
            'random_seed': config['random_state']
        }
        
        import json
        with open('results/layupformer_comparison/tsai_embedding_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\n结果已保存至 'results/layupformer_comparison/' 目录:")
        print("  - tsai_embedding_performance.png (预测散点图)")
        print("  - tsai_embedding_results.json (详细指标)")
        
    except ImportError:
        print("\n警告: 未安装matplotlib，跳过可视化")
    
    return test_r2, test_rmse, test_targets_original, test_preds_original

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置参数（严格对齐LayupFormer核心设置）
    config = {
        'data_file': 'composite_v13_dataset.csv',  # 您的数据文件
        'num_layers': 16,           # 16层铺层
        'num_layers_transformer': 4, # Transformer层数
        'd_model': 64,              # 嵌入维度
        'nhead': 8,                 # 注意力头数
        'dim_feedforward': 256,     # FFN维度
        'dropout': 0.1,             # Dropout率
        'batch_size': 64,           # 批大小
        'learning_rate': 0.001,     # 学习率
        'weight_decay': 0.01,       # 权重衰减（AdamW）
        'num_epochs': 100,          # 最大训练轮数
        'patience': 15,             # 早停耐心值
        'random_state': 42          # 随机种子
    }
    
    # 运行实验
    try:
        r2, rmse, targets, preds = evaluate_layupformer_on_v13(config)
        
        print("\n" + "="*70)
        print("LayupFormer (Tsai嵌入) 核心结论")
        print("="*70)
        print(f"✓ 严格复现Tsai不变量嵌入: Φ(θ)=[cosθ,sinθ,cos2θ,sin2θ,cos4θ,sin4θ]")
        print(f"✓ 严格80/10/10数据划分（非20%测试）")
        print(f"✓ 仅基于训练集标准化（避免数据泄露）")
        print(f"✓ 测试集R²: {r2:.6f}")
        print(f"✓ 测试集RMSE: {rmse:.6f}")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保 composite_v13_dataset.csv 文件存在于当前目录")
        print("文件应包含列: angle_1, angle_2, ..., angle_16, v13")
    
    except Exception as e:
        print(f"\n实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()