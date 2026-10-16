"""
逆序不变性消融实验
比较三种模型（Transformer, LSTM, CNN）在三种对称处理模式下的性能：
- Hard: 硬性逆序不变（正向+反向编码后平均池化）
- None: 无对称约束（仅正向编码）
- Aug:  训练时随机反转数据增强（模型同None）

输出：包含所有重复运行详细结果的Excel文件，以及聚合统计表。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import warnings
import time
from copy import deepcopy

warnings.filterwarnings('ignore')

# 设置随机种子（每个重复实验会使用不同种子，但为保证可比性，固定种子序列）
BASE_SEED = 42
N_REPEATS = 5

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 数据集（MCR编码） ====================
class MCRDataset(Dataset):
    """使用MCR编码（sin2θ, cos2θ）的数据集"""
    def __init__(self, angles, v13_scaled, lengths, num_layers):
        """
        angles: [N, L] 原始角度（度）
        v13_scaled: [N] 标准化后的v13
        lengths: [N] 序列长度（实际层数）
        num_layers: 最大层数（固定）
        """
        self.num_layers = num_layers
        # 确保角度维度正确
        if angles.shape[1] > num_layers:
            angles = angles[:, :num_layers]
        # MCR编码: sin2θ, cos2θ
        self.features = np.zeros((len(angles), num_layers, 2))
        for i in range(len(angles)):
            for j in range(num_layers):
                theta = angles[i, j]
                rad = np.radians(theta)
                self.features[i, j, 0] = np.sin(2 * rad)
                self.features[i, j, 1] = np.cos(2 * rad)
        self.v13 = v13_scaled
        self.lengths = lengths
        self.original_angles = angles

    def __len__(self):
        return len(self.v13)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.lengths[idx], dtype=torch.int64),
                torch.tensor(self.v13[idx], dtype=torch.float32),
                torch.tensor(self.original_angles[idx], dtype=torch.float32))


# ==================== 模型定义 ====================
# 1. 位置编码（共用）
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

# 2. 硬对称Transformer（正向+反向编码后平均池化）
class HardSymTransformer(nn.Module):
    """硬性逆序不变Transformer"""
    def __init__(self, input_dim=2, d_model=64, nhead=8, num_layers=4,
                 dim_feedforward=256, dropout=0.1, max_seq_len=20):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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

    def forward(self, x, lengths=None):
        # x: [batch, seq_len, 2]
        # 正向路径
        x_fwd = self.input_proj(x)
        x_fwd = self.pos_encoder(x_fwd)
        # 反向路径
        x_rev = torch.flip(x, dims=[1])
        x_rev = self.input_proj(x_rev)
        x_rev = self.pos_encoder(x_rev)
        # 分别通过Transformer（共享权重）
        fwd_out = self.transformer(x_fwd)
        rev_out = self.transformer(x_rev)
        # 平均池化
        combined = (fwd_out + rev_out) / 2
        # 全局平均池化（考虑长度掩码）
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).expand(len(lengths), -1) >= lengths.unsqueeze(1)
            valid_mask = (~mask).float().unsqueeze(-1)
            weighted_sum = torch.sum(combined * valid_mask, dim=1)
            valid_count = torch.sum(valid_mask, dim=1)
            pooled = weighted_sum / (valid_count + 1e-8)
        else:
            pooled = combined.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)

# 3. 无对称Transformer（仅正向编码）
class NoSymTransformer(nn.Module):
    """无逆序约束的Transformer"""
    def __init__(self, input_dim=2, d_model=64, nhead=8, num_layers=4,
                 dim_feedforward=256, dropout=0.1, max_seq_len=20):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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

    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).expand(len(lengths), -1) >= lengths.unsqueeze(1)
            valid_mask = (~mask).float().unsqueeze(-1)
            weighted_sum = torch.sum(x * valid_mask, dim=1)
            valid_count = torch.sum(valid_mask, dim=1)
            pooled = weighted_sum / (valid_count + 1e-8)
        else:
            pooled = x.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)

# 4. LSTM 模型（无对称版本，硬对称需特殊处理）
class NoSymLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            output_packed, (hidden, _) = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
            last_output = output[torch.arange(x.size(0)), lengths_cpu - 1]
        else:
            output, (hidden, _) = self.lstm(x)
            last_output = output[:, -1, :]
        return self.regressor(last_output).squeeze(-1)

# 硬对称LSTM：正向+反向分别LSTM后平均
class HardSymLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths=None):
        # 正向
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            x_fwd_packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            output_fwd_packed, (hidden_fwd, _) = self.lstm(x_fwd_packed)
            output_fwd, _ = nn.utils.rnn.pad_packed_sequence(output_fwd_packed, batch_first=True)
            last_fwd = output_fwd[torch.arange(x.size(0)), lengths_cpu - 1]
        else:
            output_fwd, (hidden_fwd, _) = self.lstm(x)
            last_fwd = output_fwd[:, -1, :]
        # 反向
        x_rev = torch.flip(x, dims=[1])
        if lengths is not None:
            x_rev_packed = nn.utils.rnn.pack_padded_sequence(x_rev, lengths_cpu, batch_first=True, enforce_sorted=False)
            output_rev_packed, (hidden_rev, _) = self.lstm(x_rev_packed)
            output_rev, _ = nn.utils.rnn.pad_packed_sequence(output_rev_packed, batch_first=True)
            last_rev = output_rev[torch.arange(x.size(0)), lengths_cpu - 1]
        else:
            output_rev, (hidden_rev, _) = self.lstm(x_rev)
            last_rev = output_rev[:, -1, :]
        combined = (last_fwd + last_rev) / 2
        return self.regressor(combined).squeeze(-1)

# 5. CNN 模型（无对称版本）
class NoSymCNN(nn.Module):
    def __init__(self, input_dim=2, num_filters=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(num_filters*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths=None):
        # x: [batch, seq_len, input_dim] -> [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.regressor(x).squeeze(-1)

# 硬对称CNN：正向+反向编码后平均池化
class HardSymCNN(nn.Module):
    def __init__(self, input_dim=2, num_filters=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(num_filters*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths=None):
        # 正向
        x_fwd = x.transpose(1, 2)
        fwd = torch.relu(self.conv1(x_fwd))
        fwd = torch.relu(self.conv2(fwd))
        fwd = self.pool(fwd).squeeze(-1)
        # 反向
        x_rev = torch.flip(x, dims=[1]).transpose(1, 2)
        rev = torch.relu(self.conv1(x_rev))
        rev = torch.relu(self.conv2(rev))
        rev = self.pool(rev).squeeze(-1)
        combined = (fwd + rev) / 2
        return self.regressor(combined).squeeze(-1)


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, config, use_augmentation=False):
    """通用训练函数，支持数据增强（训练时随机反转序列）"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'],
                                              epochs=config['epochs'],
                                              steps_per_epoch=len(train_loader),
                                              pct_start=0.3, anneal_strategy='cos')
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for seq, lens, targets, _ in train_loader:
            seq, targets = seq.to(device), targets.to(device)
            lens = lens.to(device)
            # 数据增强：随机反转序列（仅当启用且模型不是硬对称时）
            if use_augmentation and not isinstance(model, (HardSymTransformer, HardSymLSTM, HardSymCNN)):
                if torch.rand(1).item() < 0.5:
                    seq = torch.flip(seq, dims=[1])
            optimizer.zero_grad()
            outputs = model(seq, lens)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * seq.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, lens, targets, _ in val_loader:
                seq, targets = seq.to(device), targets.to(device)
                lens = lens.to(device)
                outputs = model(seq, lens)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seq.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())

        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{config['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate_symmetry(model, test_loader, v13_scaler):
    """评估模型：计算原始空间R²、RMSE，以及逆序预测差异"""
    model.eval()
    all_pred_fwd = []
    all_pred_rev = []
    all_target = []
    with torch.no_grad():
        for seq, lens, targets, _ in test_loader:
            seq, targets = seq.to(device), targets.to(device)
            lens = lens.to(device)
            # 正向预测
            pred_fwd = model(seq, lens).cpu().numpy()
            # 反向预测
            seq_rev = torch.flip(seq, dims=[1])
            pred_rev = model(seq_rev, lens).cpu().numpy()
            all_pred_fwd.extend(pred_fwd)
            all_pred_rev.extend(pred_rev)
            all_target.extend(targets.cpu().numpy())

    # 反归一化
    pred_fwd_orig = v13_scaler.inverse_transform(np.array(all_pred_fwd).reshape(-1,1)).flatten()
    pred_rev_orig = v13_scaler.inverse_transform(np.array(all_pred_rev).reshape(-1,1)).flatten()
    target_orig = v13_scaler.inverse_transform(np.array(all_target).reshape(-1,1)).flatten()

    r2 = r2_score(target_orig, pred_fwd_orig)
    rmse = np.sqrt(mean_squared_error(target_orig, pred_fwd_orig))
    delta = np.abs(pred_fwd_orig - pred_rev_orig)
    delta_stats = {
        'mean': np.mean(delta),
        'std': np.std(delta),
        'max': np.max(delta),
        'median': np.median(delta)
    }
    return r2, rmse, delta_stats, pred_fwd_orig, target_orig, delta


# ==================== 主实验 ====================
def load_data(file_path, num_layers=16, test_size=0.2, random_state=42):
    """加载并划分数据，返回原始角度和v13值（未标准化）"""
    df = pd.read_csv(file_path)
    angle_cols = [f'angle_{i+1}' for i in range(num_layers)]
    angles = df[angle_cols].values
    v13 = df['v13'].values
    # 划分
    train_idx, test_idx = train_test_split(np.arange(len(angles)), test_size=test_size,
                                           random_state=random_state)
    return angles, v13, train_idx, test_idx


def run_single_experiment(model_class, model_kwargs, sym_mode, train_angles, train_v13,
                          test_angles, test_v13, v13_scaler, config, run_seed):
    """运行单次实验（一个模型+一种对称模式+一个随机种子）"""
    set_seed(run_seed)
    # 准备数据加载器
    train_dataset = MCRDataset(train_angles, train_v13,
                               np.full(len(train_angles), config['num_layers']),
                               config['num_layers'])
    test_dataset = MCRDataset(test_angles, test_v13,
                              np.full(len(test_angles), config['num_layers']),
                              config['num_layers'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 创建模型
    if sym_mode == 'Hard':
        model = model_class(**model_kwargs)
        use_aug = False
    elif sym_mode == 'None':
        # 对于每种模型，需要对应的无对称版本
        if model_class.__name__ == 'HardSymTransformer':
            model = NoSymTransformer(**model_kwargs)
        elif model_class.__name__ == 'HardSymLSTM':
            model = NoSymLSTM(**model_kwargs)
        elif model_class.__name__ == 'HardSymCNN':
            model = NoSymCNN(**model_kwargs)
        else:
            raise ValueError("Unknown model class")
        use_aug = False
    elif sym_mode == 'Aug':
        # 使用无对称模型 + 数据增强
        if model_class.__name__ == 'HardSymTransformer':
            model = NoSymTransformer(**model_kwargs)
        elif model_class.__name__ == 'HardSymLSTM':
            model = NoSymLSTM(**model_kwargs)
        elif model_class.__name__ == 'HardSymCNN':
            model = NoSymCNN(**model_kwargs)
        else:
            raise ValueError("Unknown model class")
        use_aug = True
    else:
        raise ValueError("sym_mode must be Hard/None/Aug")

    # 训练
    model, _ = train_model(model, train_loader, test_loader, config, use_augmentation=use_aug)
    # 评估
    r2, rmse, delta_stats, _, _, _ = evaluate_symmetry(model, test_loader, v13_scaler)
    return {
        'r2': r2,
        'rmse': rmse,
        'delta_mean': delta_stats['mean'],
        'delta_std': delta_stats['std'],
        'delta_max': delta_stats['max'],
        'delta_median': delta_stats['median']
    }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 配置
    config = {
        'data_file': 'composite_v13_dataset.csv',
        'num_layers': 16,
        'test_size': 0.2,
        'batch_size': 64,
        'lr': 0.001,
        'epochs': 80,
        'd_model': 64,
        'nhead': 8,
        'num_layers_trans': 4,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'hidden_dim_lstm': 128,
        'num_layers_lstm': 2,
        'cnn_filters': 64,
        'cnn_kernel': 3
    }

    # 模型定义（硬对称版本，用于构造）
    model_classes = {
        'Transformer': (HardSymTransformer, {
            'input_dim': 2, 'd_model': config['d_model'], 'nhead': config['nhead'],
            'num_layers': config['num_layers_trans'], 'dim_feedforward': config['dim_feedforward'],
            'dropout': config['dropout'], 'max_seq_len': config['num_layers']
        }),
        'LSTM': (HardSymLSTM, {
            'input_dim': 2, 'hidden_dim': config['hidden_dim_lstm'],
            'num_layers': config['num_layers_lstm'], 'dropout': config['dropout']
        }),
        'CNN': (HardSymCNN, {
            'input_dim': 2, 'num_filters': config['cnn_filters'],
            'kernel_size': config['cnn_kernel'], 'dropout': config['dropout']
        })
    }

    sym_modes = ['Hard', 'None', 'Aug']

    # 加载数据（全局，固定划分）
    print("Loading data...")
    angles, v13, train_idx, test_idx = load_data(config['data_file'], config['num_layers'],
                                                 config['test_size'], random_state=42)
    # 标准化v13（基于训练集）
    v13_scaler = StandardScaler()
    train_v13_scaled = v13_scaler.fit_transform(v13[train_idx].reshape(-1,1)).flatten()
    test_v13_scaled = v13_scaler.transform(v13[test_idx].reshape(-1,1)).flatten()

    train_angles = angles[train_idx]
    test_angles = angles[test_idx]

    # 存储所有详细结果
    all_results = []

    for model_name, (model_class, model_kwargs) in model_classes.items():
        for sym_mode in sym_modes:
            print(f"\n{'='*60}")
            print(f"Running: {model_name} - {sym_mode}")
            print(f"{'='*60}")
            repeat_results = []
            for rep in range(N_REPEATS):
                seed = BASE_SEED + rep * 10 + hash(model_name + sym_mode) % 1000
                print(f"  Repeat {rep+1}/{N_REPEATS} (seed={seed})")
                res = run_single_experiment(model_class, model_kwargs, sym_mode,
                                            train_angles, train_v13_scaled,
                                            test_angles, test_v13_scaled,
                                            v13_scaler, config, seed)
                res['repeat'] = rep + 1
                res['model'] = model_name
                res['sym_mode'] = sym_mode
                repeat_results.append(res)
                all_results.append(res)
            # 打印当前组合的聚合统计
            df_tmp = pd.DataFrame(repeat_results)
            print(f"  Summary: R² = {df_tmp['r2'].mean():.4f} ± {df_tmp['r2'].std():.4f} | "
                  f"RMSE = {df_tmp['rmse'].mean():.6f} ± {df_tmp['rmse'].std():.6f} | "
                  f"Δ_mean = {df_tmp['delta_mean'].mean():.6f}")

    # 保存所有详细结果
    df_all = pd.DataFrame(all_results)
    df_all.to_excel('ablation_reversal_detailed.xlsx', index=False)
    print("\n详细结果已保存至 ablation_reversal_detailed.xlsx")

    # 生成聚合统计表（用于论文）
    agg = df_all.groupby(['model', 'sym_mode']).agg({
        'r2': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'delta_mean': ['mean', 'std'],
        'delta_std': ['mean', 'std'],
        'delta_max': ['mean', 'std'],
        'delta_median': ['mean', 'std']
    }).round(6)
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index().to_excel('ablation_reversal_aggregated.xlsx', index=False)
    print("聚合统计表已保存至 ablation_reversal_aggregated.xlsx")

    print("\n实验完成！")


if __name__ == "__main__":
    main()