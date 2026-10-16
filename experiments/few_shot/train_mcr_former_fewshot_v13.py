import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 导入 MCR_Former 模块
# ---------------------------------------------------------
sys.path.append('.')
try:
    from MCR_Former import MCR_Former
    print("Successfully imported MCR_Former")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure MCR_Former.py is in the same directory")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. 数据集类：复合材料层压板 (Composite Laminate)
# ---------------------------------------------------------
class CompositeLaminateDataset(Dataset):
    def __init__(self, angles_df, v13_series, max_layers):
        """
        angles_df: DataFrame, columns are angle_1, angle_2, ..., angle_N
        v13_series: Series, target values
        max_layers: int, maximum number of layers in the dataset (for padding)
        """
        self.max_layers = max_layers
        self.num_samples = len(angles_df)
        
        # 初始化特征张量 [num_samples, max_layers, 2] for (sin2θ, cos2θ)
        self.features = np.zeros((self.num_samples, self.max_layers, 2), dtype=np.float32)
        self.lengths = np.zeros(self.num_samples, dtype=np.int64)
        self.targets = v13_series.values.astype(np.float32)
        
        # 获取角度列名并排序
        angle_cols = sorted([col for col in angles_df.columns if col.startswith('angle')], 
                            key=lambda x: int(x.split('_')[1]))
        
        for i in range(self.num_samples):
            row = angles_df.iloc[i]
            # 确定实际层数
            actual_layers = len(angle_cols)
            self.lengths[i] = actual_layers
            
            for j in range(actual_layers):
                angle_deg = row[angle_cols[j]]
                angle_rad = np.deg2rad(angle_deg)
                
                # MCR Encoding for 180-degree symmetry: sin(2θ), cos(2θ)
                self.features[i, j, 0] = np.sin(2 * angle_rad)
                self.features[i, j, 1] = np.cos(2 * angle_rad)
                
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.lengths[idx], dtype=torch.int64),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# ---------------------------------------------------------
# 3. 数据加载与预处理
# ---------------------------------------------------------
def load_composite_data(data_dir, min_layer=10, max_layer=25):
    """
    Load all layer_X.csv files from data_dir within [min_layer, max_layer].
    Returns combined DataFrames for angles and targets.
    """
    all_angles_dfs = []
    all_targets = []
    all_layer_counts = []
    
    print(f"Loading data from {data_dir} for layers {min_layer} to {max_layer}...")
    
    for layer_count in range(min_layer, max_layer + 1):
        filename = f"layer_{layer_count}.csv"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue
            
        df = pd.read_csv(filepath)
        
        # 假设最后一列是 v13，或者列名为 'v13' / 'target'
        target_col = None
        for col in df.columns:
            if 'v13' in col.lower() or 'target' in col.lower():
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]
            
        targets = df[target_col]
        angles = df.drop(columns=[target_col])
        
        all_angles_dfs.append(angles)
        all_targets.append(targets)
        all_layer_counts.extend([layer_count] * len(df))
        
        print(f"Loaded {filename}: {len(df)} samples, {len(angles.columns)} layers")
        
    if not all_angles_dfs:
        raise FileNotFoundError("No data files found in the specified range.")
        
    combined_angles = pd.concat(all_angles_dfs, ignore_index=True)
    combined_targets = pd.concat(all_targets, ignore_index=True)
    
    return combined_angles, combined_targets, all_layer_counts

def prepare_few_shot_split(pretrain_angles, pretrain_targets, pretrain_layers, 
                           target_layer_num, n_shots=10, random_state=42):
    """
    Split data into:
    1. Pre-training set: All layers except target_layer_num
    2. Few-shot target set: n_shots samples from target_layer_num
    3. Test set: Remaining samples from target_layer_num
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Identify indices for target layer and other layers
    target_indices = [i for i, l in enumerate(pretrain_layers) if l == target_layer_num]
    other_indices = [i for i, l in enumerate(pretrain_layers) if l != target_layer_num]
    
    if len(target_indices) < n_shots + 10: # Ensure enough for test
        raise ValueError(f"Not enough samples for layer {target_layer_num}. Found {len(target_indices)}, need at least {n_shots + 10}.")
        
    # 2. Shuffle target indices
    rng.shuffle(target_indices)
    few_shot_train_indices = target_indices[:n_shots]
    test_indices = target_indices[n_shots:]
    
    # 3. Construct DataFrames/Series for each split
    # Pre-train data (all other layers)
    pt_angles = pretrain_angles.iloc[other_indices].reset_index(drop=True)
    pt_targets = pretrain_targets.iloc[other_indices].reset_index(drop=True)
    pt_layers = [pretrain_layers[i] for i in other_indices]
    
    # Few-shot train data
    fs_angles = pretrain_angles.iloc[few_shot_train_indices].reset_index(drop=True)
    fs_targets = pretrain_targets.iloc[few_shot_train_indices].reset_index(drop=True)
    fs_layers = [pretrain_layers[i] for i in few_shot_train_indices]
    
    # Test data
    test_angles = pretrain_angles.iloc[test_indices].reset_index(drop=True)
    test_targets = pretrain_targets.iloc[test_indices].reset_index(drop=True)
    test_layers = [pretrain_layers[i] for i in test_indices]
    
    return (pt_angles, pt_targets, pt_layers), \
           (fs_angles, fs_targets, fs_layers), \
           (test_angles, test_targets, test_layers)

# ---------------------------------------------------------
# 4. 训练与评估函数
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, clip_value=1.0):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    model_saved = False
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_train_loss = 0
        nan_detected = False
        
        for features, lengths, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            features, targets = features.to(device), targets.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(features, lengths)
            loss = criterion(outputs.squeeze(), targets)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at Epoch {epoch+1}. Stopping training.")
                nan_detected = True
                break
                
            loss.backward()
            
            # Gradient Clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            epoch_train_loss += loss.item() * features.size(0)
            
        if nan_detected:
            break
            
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for features, lengths, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                lengths = lengths.to(device)
                outputs = model(features, lengths)
                loss = criterion(outputs.squeeze(), targets)
                epoch_val_loss += loss.item() * features.size(0)
                
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best state temporarily
            torch.save(model.state_dict(), 'best_pretrained_model_temp.pth')
            model_saved = True
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
    if not model_saved:
        print("Warning: No model was saved during training (likely due to NaN or poor performance). Saving current state.")
        torch.save(model.state_dict(), 'best_pretrained_model_temp.pth')
        
    # Load best model
    model.load_state_dict(torch.load('best_pretrained_model_temp.pth'))
    return train_losses, val_losses

def evaluate_model(model, data_loader, target_scaler):
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = []
    
    with torch.no_grad():
        for features, lengths, targets in data_loader:
            features = features.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            
            outputs = model(features, lengths)
            
            all_preds_scaled.extend(outputs.cpu().numpy().flatten())
            all_targets_scaled.extend(targets.cpu().numpy().flatten())
            
    all_preds_scaled = np.array(all_preds_scaled).reshape(-1, 1)
    all_targets_scaled = np.array(all_targets_scaled).reshape(-1, 1)
    
    # Inverse transform to original scale
    all_preds_original = target_scaler.inverse_transform(all_preds_scaled)
    all_targets_original = target_scaler.inverse_transform(all_targets_scaled)
    
    r2 = r2_score(all_targets_original, all_preds_original)
    rmse = np.sqrt(mean_squared_error(all_targets_original, all_preds_original))
    return r2, rmse, all_preds_original.flatten(), all_targets_original.flatten()

# ---------------------------------------------------------
# 5. 主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    DATA_DIR = 'data'
    PRETRAIN_MIN_LAYER = 10
    PRETRAIN_MAX_LAYER = 20
    TARGET_LAYER = 21
    N_SHOTS = 10
    MAX_LAYERS_GLOBAL = 25 # Padding size
    
    BATCH_SIZE_PRETRAIN = 64
    BATCH_SIZE_FEWSHOT = 10 # Small batch for few-shot
    NUM_EPOCHS_PRETRAIN = 30
    NUM_EPOCHS_FEWSHOT = 50 # Few-shot needs fewer epochs to avoid overfitting
    
    LR_PRETRAIN = 1e-4 # Reduced from 1e-3 to prevent NaN
    LR_FEWSHOT = 1e-5 # Lower LR for fine-tuning
    
    # --- Step 1: Load All Data (10-25 layers) ---
    all_angles, all_targets, all_layers = load_composite_data(DATA_DIR, PRETRAIN_MIN_LAYER, 25)
    
    # --- Step 2: Split into Pre-train and Target (21-layer) sets ---
    (pt_angles, pt_targets, pt_layers), \
    (fs_angles, fs_targets, fs_layers), \
    (test_angles, test_targets, test_layers) = prepare_few_shot_split(
        all_angles, all_targets, all_layers, 
        target_layer_num=TARGET_LAYER, 
        n_shots=N_SHOTS
    )
    
    print(f"\nData Split Summary:")
    print(f"Pre-train layers ({PRETRAIN_MIN_LAYER}-{PRETRAIN_MAX_LAYER}): {len(pt_angles)} samples")
    print(f"Few-shot train (Layer {TARGET_LAYER}): {len(fs_angles)} samples")
    print(f"Test (Layer {TARGET_LAYER}): {len(test_angles)} samples")
    
    # --- Step 3: Create Datasets & Loaders ---
    # Determine max layers in pre-train set for padding
    max_pt_layers = max(pt_layers) if pt_layers else MAX_LAYERS_GLOBAL
    
    pt_dataset = CompositeLaminateDataset(pt_angles, pt_targets, max_pt_layers)
    fs_dataset = CompositeLaminateDataset(fs_angles, fs_targets, MAX_LAYERS_GLOBAL)
    test_dataset = CompositeLaminateDataset(test_angles, test_targets, MAX_LAYERS_GLOBAL)
    
    # Split pre-train into train/val
    pt_indices = np.arange(len(pt_dataset))
    pt_train_idx, pt_val_idx = train_test_split(pt_indices, test_size=0.1, random_state=42)
    
    pt_train_subset = torch.utils.data.Subset(pt_dataset, pt_train_idx)
    pt_val_subset = torch.utils.data.Subset(pt_dataset, pt_val_idx)
    
    pt_train_loader = DataLoader(pt_train_subset, batch_size=BATCH_SIZE_PRETRAIN, shuffle=True)
    pt_val_loader = DataLoader(pt_val_subset, batch_size=BATCH_SIZE_PRETRAIN, shuffle=False)
    
    fs_loader = DataLoader(fs_dataset, batch_size=BATCH_SIZE_FEWSHOT, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_FEWSHOT, shuffle=False)
    
    # --- Target Scaling (Crucial for stability) ---
    target_scaler = StandardScaler()
    # Fit scaler ONLY on pre-training targets to avoid data leakage from test/few-shot
    target_scaler.fit(pt_targets.values.reshape(-1, 1))
    
    # Create scaled datasets for training
    class ScaledCompositeDataset(Dataset):
        def __init__(self, base_dataset, scaler):
            self.base_dataset = base_dataset
            self.scaler = scaler
            
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            features, lengths, target = self.base_dataset[idx]
            scaled_target = self.scaler.transform(target.numpy().reshape(1, -1))[0][0]
            return features, lengths, torch.tensor(scaled_target, dtype=torch.float32)

    pt_train_scaled = ScaledCompositeDataset(pt_train_subset, target_scaler)
    pt_val_scaled = ScaledCompositeDataset(pt_val_subset, target_scaler)
    
    # Re-create loaders with scaled targets
    pt_train_loader = DataLoader(pt_train_scaled, batch_size=BATCH_SIZE_PRETRAIN, shuffle=True)
    pt_val_loader = DataLoader(pt_val_scaled, batch_size=BATCH_SIZE_PRETRAIN, shuffle=False)
    
    # For few-shot and test, we also need scaled targets during training/eval
    fs_scaled = ScaledCompositeDataset(fs_dataset, target_scaler)
    test_scaled = ScaledCompositeDataset(test_dataset, target_scaler)
    
    fs_loader = DataLoader(fs_scaled, batch_size=BATCH_SIZE_FEWSHOT, shuffle=True)
    test_loader = DataLoader(test_scaled, batch_size=BATCH_SIZE_FEWSHOT, shuffle=False)


    # --- Step 4: Pre-training on Layers 10-20 ---
    print("\n" + "="*60)
    print("PHASE 1: Pre-training on Layers 10-20")
    print("="*60)
    
    model = MCR_Former(
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=MAX_LAYERS_GLOBAL
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_PRETRAIN)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    start_time = time.time()
    try:
        train_losses, val_losses = train_model(
            model, pt_train_loader, pt_val_loader, 
            criterion, optimizer, scheduler, 
            num_epochs=NUM_EPOCHS_PRETRAIN,
            clip_value=1.0 # Add gradient clipping
        )
        pretrain_time = time.time() - start_time
        print(f"Pre-training completed in {pretrain_time:.2f}s")
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
    
    # Evaluate Zero-shot performance on Layer 21 (Before Fine-tuning)
    r2_zero, rmse_zero, _, _ = evaluate_model(model, test_loader, target_scaler)
    print(f"\nZero-shot Performance on Layer {TARGET_LAYER} (Before Fine-tuning):")
    print(f"R²: {r2_zero:.4f}, RMSE: {rmse_zero:.6f}")
    
    # --- Step 5: Few-Shot Fine-tuning on Layer 21 (10 samples) ---
    print("\n" + "="*60)
    print(f"PHASE 2: Few-Shot Fine-tuning on Layer {TARGET_LAYER} ({N_SHOTS} samples)")
    print("="*60)
    
    # Reset optimizer for fine-tuning with lower learning rate
    optimizer_fs = optim.Adam(model.parameters(), lr=LR_FEWSHOT)
    scheduler_fs = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fs, 'min', patience=5, factor=0.5)
    
    model.train()
    for epoch in range(NUM_EPOCHS_FEWSHOT):
        epoch_loss = 0
        for features, lengths, targets in fs_loader:
            features, targets = features.to(device), targets.to(device)
            lengths = lengths.to(device)
            
            optimizer_fs.zero_grad()
            outputs = model(features, lengths)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            
            # Gradient clipping for fine-tuning as well
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value=1.0)
            
            optimizer_fs.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Fine-tune Epoch {epoch+1}/{NUM_EPOCHS_FEWSHOT}: Loss: {epoch_loss/len(fs_loader):.6f}")
            
    # --- Step 6: Final Evaluation ---
    r2_final, rmse_final, preds, targets_true = evaluate_model(model, test_loader, target_scaler)
    
    print(f"\nFinal Performance on Layer {TARGET_LAYER} (After {N_SHOTS}-shot Fine-tuning):")
    print(f"R²: {r2_final:.4f}, RMSE: {rmse_final:.6f}")
    print(f"Improvement in R²: {r2_final - r2_zero:.4f}")
    
    # --- Step 7: Visualization ---
    plt.figure(figsize=(10, 6))
    plt.scatter(targets_true, preds, alpha=0.7, color='green', label='Predicted vs Actual')
    min_val = min(targets_true.min(), preds.min())
    max_val = max(targets_true.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual v13')
    plt.ylabel('Predicted v13')
    plt.title(f'Few-Shot Result: Layer {TARGET_LAYER} ({N_SHOTS} shots)\nR²={r2_final:.4f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'results_fewshot_layer{TARGET_LAYER}_{N_SHOTS}shots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nResults saved to results_fewshot_layer*.png")