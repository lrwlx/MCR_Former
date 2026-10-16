import torch
import numpy as np
import pandas as pd
import math
import time
import json
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# 导入计算真实v13的函数
try:
    from get_v13 import get_v13
    print("成功导入get_v13模块")
except ImportError:
    # 如果导入失败，创建一个占位函数
    def get_v13(angles):
        print(f"警告: 使用占位函数计算v13，实际应使用真实物理公式")
        return np.random.uniform(-0.5, 0.5)  # 返回随机值作为占位
    
    print("警告: 未能导入get_v13模块，使用占位函数")

# ==================== 数据准备函数 ====================
def prepare_data_with_mcr(file_path='composite_v13_dataset_8.csv', num_layers=8, test_size=0.2, random_state=42):
    """准备训练和测试数据（使用MCR特征）"""
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # 提取角度数据
    angle_columns = [f'angle_{i+1}' for i in range(num_layers)]
    angles = df[angle_columns].values
    
    # 提取v13目标值
    v13_values = df['v13'].values
    
    # 转换为MCR特征 (sin2θ, cos2θ)
    angle_features = np.zeros((len(angles), num_layers, 2))
    for i in range(len(angles)):
        for j in range(num_layers):
            theta = angles[i, j]
            rad = np.radians(theta)
            angle_features[i, j, 0] = np.sin(2 * rad)
            angle_features[i, j, 1] = np.cos(2 * rad)
    
    # 划分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        angle_features, v13_values, test_size=test_size, random_state=random_state
    )
    
    # 为SVR和MLP展平数据 (样本数, num_layers*2)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 为LSTM保持序列结构 (样本数, 序列长度, 特征数)
    X_train_seq = X_train
    X_test_seq = X_test
    
    return {
        'angles': angles,
        'v13_values': v13_values,
        'X_train_flat': X_train_flat,
        'X_test_flat': X_test_flat,
        'X_train_seq': X_train_seq,
        'X_test_seq': X_test_seq,
        'y_train': y_train,
        'y_test': y_test,
        'num_layers': num_layers,
        'feature_type': 'mcr'
    }

def prepare_data_without_mcr(file_path='composite_v13_dataset_8.csv', num_layers=8, test_size=0.2, random_state=42):
    """准备训练和测试数据（不使用MCR特征）"""
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # 提取角度数据
    angle_columns = [f'angle_{i+1}' for i in range(num_layers)]
    angles = df[angle_columns].values
    
    # 提取v13目标值
    v13_values = df['v13'].values
    
    # 注意：这里我们直接使用原始角度，而不进行MCR变换
    angle_features = angles.reshape(len(angles), num_layers, 1)  # 每个角度作为一个特征
    
    # 划分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        angle_features, v13_values, test_size=test_size, random_state=random_state
    )
    
    # 为SVR和MLP展平数据 (样本数, num_layers)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 为LSTM保持序列结构 (样本数, 序列长度, 特征数=1)
    X_train_seq = X_train
    X_test_seq = X_test
    
    return {
        'angles': angles,
        'v13_values': v13_values,
        'X_train_flat': X_train_flat,
        'X_test_flat': X_test_flat,
        'X_train_seq': X_train_seq,
        'X_test_seq': X_test_seq,
        'y_train': y_train,
        'y_test': y_test,
        'num_layers': num_layers,
        'feature_type': 'raw_angles'
    }

# ==================== SVR模型 ====================
class SVR_Model:
    """支持向量回归模型"""
    def __init__(self, data, kernel='rbf', C=1.0, epsilon=0.1, use_mcr=True):
        self.data = data
        self.use_mcr = use_mcr
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def train(self):
        """训练SVR模型"""
        # 归一化输入特征
        X_train_scaled = self.scaler_X.fit_transform(self.data['X_train_flat'])
        y_train_scaled = self.scaler_y.fit_transform(self.data['y_train'].reshape(-1, 1)).flatten()
        
        model_name = "SVR" if self.use_mcr else "SVR (No MCR)"
        print(f"Training {model_name} model...")
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train_scaled)
        train_time = time.time() - start_time
        print(f"{model_name} training completed in {train_time:.2f} seconds")
        
        # 评估模型
        X_test_scaled = self.scaler_X.transform(self.data['X_test_flat'])
        y_test_scaled = self.scaler_y.transform(self.data['y_test'].reshape(-1, 1)).flatten()
        y_pred_scaled = self.model.predict(X_test_scaled)
        
        # 反归一化
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test = self.data['y_test']
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{model_name} Test R²: {r2:.4f}, RMSE: {rmse:.6f}")
        
        return r2, rmse
        
    def predict(self, X):
        """预测v13值"""
        if X.ndim == 3:  # 如果是序列数据，展平
            X = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def inverse_design(self, num_restarts=3, maxiter=50, popsize=15):
        """使用SVR模型进行逆向设计"""
        bounds = [(-90, 90) for _ in range(self.data['num_layers'])]
        
        model_name = "SVR" if self.use_mcr else "SVR (No MCR)"
        print(f"\n" + "="*60)
        print(f"{model_name} Inverse Design using Differential Evolution")
        print("="*60)
        
        best_v13 = float('inf')
        best_angles = None
        best_result = None
        
        def objective(angles):
            if self.use_mcr:
                # 将角度转换为MCR特征
                angles_rad = np.radians(angles)
                features = np.zeros((1, self.data['num_layers'], 2))
                for j in range(self.data['num_layers']):
                    features[0, j, 0] = np.sin(2 * angles_rad[j])
                    features[0, j, 1] = np.cos(2 * angles_rad[j])
            else:
                # 直接使用原始角度作为特征
                features = angles.reshape(1, -1)
            
            # 预测v13
            v13_pred = self.predict(features)[0]
            return v13_pred
        
        for restart in range(num_restarts):
            print(f"\nRestart #{restart+1}/{num_restarts}")
            start_time = time.time()
            
            result = differential_evolution(
                objective,
                bounds,
                strategy='best1bin',
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-6,
                polish=True,
                disp=False
            )
            
            elapsed_time = time.time() - start_time
            print(f"  Found v13 = {result.fun:.6f}, Time: {elapsed_time:.2f}s")
            
            if result.fun < best_v13:
                best_v13 = result.fun
                best_angles = result.x
                best_result = result
        
        # 计算真实v13值
        true_v13 = get_v13(best_angles.tolist())
        
        print(f"\n{model_name} Best Result:")
        print(f"  Predicted v13: {best_v13:.6f}")
        print(f"  True v13: {true_v13:.6f}")
        print(f"  Prediction Error: {abs(best_v13 - true_v13):.6f}")
        print(f"  Optimal Angles: {np.round(best_angles, 2)}")
        
        return best_v13, true_v13, best_angles

# ==================== MLP模型 ====================
class MLP_Model:
    """多层感知机模型"""
    def __init__(self, data, hidden_layers=(128, 64, 32), activation='relu', learning_rate=0.001, max_iter=500, use_mcr=True):
        self.data = data
        self.use_mcr = use_mcr
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def train(self):
        """训练MLP模型"""
        # 归一化输入特征
        X_train_scaled = self.scaler_X.fit_transform(self.data['X_train_flat'])
        y_train_scaled = self.scaler_y.fit_transform(self.data['y_train'].reshape(-1, 1)).flatten()
        
        model_name = "MLP" if self.use_mcr else "MLP (No MCR)"
        print(f"\nTraining {model_name} model...")
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train_scaled)
        train_time = time.time() - start_time
        print(f"{model_name} training completed in {train_time:.2f} seconds")
        print(f"Training iterations: {self.model.n_iter_}")
        
        # 评估模型
        X_test_scaled = self.scaler_X.transform(self.data['X_test_flat'])
        y_test_scaled = self.scaler_y.transform(self.data['y_test'].reshape(-1, 1)).flatten()
        y_pred_scaled = self.model.predict(X_test_scaled)
        
        # 反归一化
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test = self.data['y_test']
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{model_name} Test R²: {r2:.4f}, RMSE: {rmse:.6f}")
        
        return r2, rmse
        
    def predict(self, X):
        """预测v13值"""
        if X.ndim == 3:  # 如果是序列数据，展平
            X = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def inverse_design(self, num_restarts=3, maxiter=50, popsize=15):
        """使用MLP模型进行逆向设计"""
        bounds = [(-90, 90) for _ in range(self.data['num_layers'])]
        
        model_name = "MLP" if self.use_mcr else "MLP (No MCR)"
        print(f"\n" + "="*60)
        print(f"{model_name} Inverse Design using Differential Evolution")
        print("="*60)
        
        best_v13 = float('inf')
        best_angles = None
        
        def objective(angles):
            if self.use_mcr:
                # 将角度转换为MCR特征
                angles_rad = np.radians(angles)
                features = np.zeros((1, self.data['num_layers'], 2))
                for j in range(self.data['num_layers']):
                    features[0, j, 0] = np.sin(2 * angles_rad[j])
                    features[0, j, 1] = np.cos(2 * angles_rad[j])
            else:
                # 直接使用原始角度作为特征
                features = angles.reshape(1, -1)
            
            # 预测v13
            v13_pred = self.predict(features)[0]
            return v13_pred
        
        for restart in range(num_restarts):
            print(f"\nRestart #{restart+1}/{num_restarts}")
            start_time = time.time()
            
            result = differential_evolution(
                objective,
                bounds,
                strategy='best1bin',
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-6,
                polish=True,
                disp=False
            )
            
            elapsed_time = time.time() - start_time
            print(f"  Found v13 = {result.fun:.6f}, Time: {elapsed_time:.2f}s")
            
            if result.fun < best_v13:
                best_v13 = result.fun
                best_angles = result.x
        
        # 计算真实v13值
        true_v13 = get_v13(best_angles.tolist())
        
        print(f"\n{model_name} Best Result:")
        print(f"  Predicted v13: {best_v13:.6f}")
        print(f"  True v13: {true_v13:.6f}")
        print(f"  Prediction Error: {abs(best_v13 - true_v13):.6f}")
        print(f"  Optimal Angles: {np.round(best_angles, 2)}")
        
        return best_v13, true_v13, best_angles

# ==================== LSTM模型 ====================
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,  # 增加dropout防止过拟合
            bidirectional=False
        )
        # 减小全连接层大小
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)

class LSTMModel_NoMCR(nn.Module):
    """LSTM模型（无MCR）"""
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        # 减小全连接层大小
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)

class LSTM_Model:
    """LSTM模型包装类"""
    def __init__(self, data, hidden_size=32, num_layers=1, learning_rate=0.001, num_epochs=50, use_mcr=True):
        self.data = data
        self.use_mcr = use_mcr
        
        if use_mcr:
            self.model = LSTMModel(input_size=2, hidden_size=hidden_size, 
                                  num_layers=num_layers, output_size=1)
        else:
            self.model = LSTMModel_NoMCR(input_size=1, hidden_size=hidden_size, 
                                        num_layers=num_layers, output_size=1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.scaler_y = StandardScaler()
        self.num_epochs = num_epochs  # 减少训练轮数
        self.learning_rate = learning_rate
        
    def train(self):
        """训练LSTM模型"""
        # 准备数据
        X_train = torch.tensor(self.data['X_train_seq'], dtype=torch.float32)
        y_train = torch.tensor(self.data['y_train'], dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(self.data['X_test_seq'], dtype=torch.float32)
        y_test = torch.tensor(self.data['y_test'], dtype=torch.float32).view(-1, 1)
        
        # 归一化目标值
        y_train_scaled = torch.tensor(
            self.scaler_y.fit_transform(y_train.numpy()), dtype=torch.float32
        ).squeeze()
        y_test_scaled = torch.tensor(
            self.scaler_y.transform(y_test.numpy()), dtype=torch.float32
        ).squeeze()
        
        # 创建数据加载器（使用更小的批大小）
        train_dataset = TensorDataset(X_train, y_train_scaled)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 从32减少到16
        
        # 训练模型
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        model_name = "LSTM" if self.use_mcr else "LSTM (No MCR)"
        print(f"\nTraining {model_name} model...")
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 减少输出频率
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = train_loss / len(train_loader)
                print(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
        
        train_time = time.time() - start_time
        print(f"{model_name} training completed in {train_time:.2f} seconds")
        
        # 评估模型
        self.model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(self.device)
            y_pred_scaled = self.model(X_test_device).cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            r2 = r2_score(self.data['y_test'], y_pred)
            rmse = np.sqrt(mean_squared_error(self.data['y_test'], y_pred))
            print(f"{model_name} Test R²: {r2:.4f}, RMSE: {rmse:.6f}")
        
        return r2, rmse
        
    def predict(self, X):
        """预测v13值"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = X.to(self.device)
            
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
            return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    def inverse_design(self, num_restarts=3, maxiter=50, popsize=15):
        """使用LSTM模型进行逆向设计"""
        bounds = [(-90, 90) for _ in range(self.data['num_layers'])]
        
        model_name = "LSTM" if self.use_mcr else "LSTM (No MCR)"
        print(f"\n" + "="*60)
        print(f"{model_name} Inverse Design using Differential Evolution")
        print("="*60)
        
        best_v13 = float('inf')
        best_angles = None
        
        def objective(angles):
            if self.use_mcr:
                # 将角度转换为MCR特征
                angles_rad = np.radians(angles)
                features = np.zeros((1, self.data['num_layers'], 2))
                for j in range(self.data['num_layers']):
                    features[0, j, 0] = np.sin(2 * angles_rad[j])
                    features[0, j, 1] = np.cos(2 * angles_rad[j])
            else:
                # 直接使用原始角度作为特征
                features = angles.reshape(1, -1, 1)  # (batch_size, seq_len, input_size)
            
            # 转换为张量并预测
            features_tensor = torch.tensor(features, dtype=torch.float32)
            v13_pred = self.predict(features_tensor)[0]
            return v13_pred
        
        for restart in range(num_restarts):
            print(f"\nRestart #{restart+1}/{num_restarts}")
            start_time = time.time()
            
            result = differential_evolution(
                objective,
                bounds,
                strategy='best1bin',
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-6,
                polish=True,
                disp=False
            )
            
            elapsed_time = time.time() - start_time
            print(f"  Found v13 = {result.fun:.6f}, Time: {elapsed_time:.2f}s")
            
            if result.fun < best_v13:
                best_v13 = result.fun
                best_angles = result.x
        
        # 计算真实v13值
        true_v13 = get_v13(best_angles.tolist())
        
        print(f"\n{model_name} Best Result:")
        print(f"  Predicted v13: {best_v13:.6f}")
        print(f"  True v13: {true_v13:.6f}")
        print(f"  Prediction Error: {abs(best_v13 - true_v13):.6f}")
        print(f"  Optimal Angles: {np.round(best_angles, 2)}")
        
        return best_v13, true_v13, best_angles

# ==================== 主程序 ====================
def compare_all_models():
    """比较所有模型（整合版本）"""
    print("="*70)
    print("COMPARATIVE STUDY: All Models With and Without MCR")
    print("="*70)
    
    # 准备数据
    data_with_mcr = prepare_data_with_mcr()
    data_without_mcr = prepare_data_without_mcr()
    
    # 存储结果
    results = {}
    
    # 1. SVR模型（有MCR）
    print("\n" + "="*60)
    print("1. SUPPORT VECTOR REGRESSION (SVR) - With MCR")
    print("="*60)
    svr_model_mcr = SVR_Model(data_with_mcr, use_mcr=True)
    svr_r2_mcr, svr_rmse_mcr = svr_model_mcr.train()
    svr_pred_v13_mcr, svr_true_v13_mcr, svr_angles_mcr = svr_model_mcr.inverse_design()
    results['SVR (MCR)'] = {
        'test_r2': svr_r2_mcr,
        'test_rmse': svr_rmse_mcr,
        'predicted_v13': svr_pred_v13_mcr,
        'true_v13': svr_true_v13_mcr,
        'prediction_error': abs(svr_pred_v13_mcr - svr_true_v13_mcr),
        'optimal_angles': svr_angles_mcr.tolist(),
        'feature_type': 'MCR'
    }
    
    # 2. SVR模型（无MCR）
    print("\n" + "="*60)
    print("2. SUPPORT VECTOR REGRESSION (SVR) - Without MCR")
    print("="*60)
    svr_model_no_mcr = SVR_Model(data_without_mcr, use_mcr=False)
    svr_r2_no_mcr, svr_rmse_no_mcr = svr_model_no_mcr.train()
    svr_pred_v13_no_mcr, svr_true_v13_no_mcr, svr_angles_no_mcr = svr_model_no_mcr.inverse_design()
    results['SVR (No MCR)'] = {
        'test_r2': svr_r2_no_mcr,
        'test_rmse': svr_rmse_no_mcr,
        'predicted_v13': svr_pred_v13_no_mcr,
        'true_v13': svr_true_v13_no_mcr,
        'prediction_error': abs(svr_pred_v13_no_mcr - svr_true_v13_no_mcr),
        'optimal_angles': svr_angles_no_mcr.tolist(),
        'feature_type': 'Raw Angles'
    }
    
    # 3. MLP模型（有MCR）
    print("\n" + "="*60)
    print("3. MULTI-LAYER PERCEPTRON (MLP) - With MCR")
    print("="*60)
    mlp_model_mcr = MLP_Model(data_with_mcr, hidden_layers=(256, 128, 64), max_iter=300, use_mcr=True)
    mlp_r2_mcr, mlp_rmse_mcr = mlp_model_mcr.train()
    mlp_pred_v13_mcr, mlp_true_v13_mcr, mlp_angles_mcr = mlp_model_mcr.inverse_design()
    results['MLP (MCR)'] = {
        'test_r2': mlp_r2_mcr,
        'test_rmse': mlp_rmse_mcr,
        'predicted_v13': mlp_pred_v13_mcr,
        'true_v13': mlp_true_v13_mcr,
        'prediction_error': abs(mlp_pred_v13_mcr - mlp_true_v13_mcr),
        'optimal_angles': mlp_angles_mcr.tolist(),
        'feature_type': 'MCR'
    }
    
    # 4. MLP模型（无MCR）
    print("\n" + "="*60)
    print("4. MULTI-LAYER PERCEPTRON (MLP) - Without MCR")
    print("="*60)
    mlp_model_no_mcr = MLP_Model(data_without_mcr, hidden_layers=(256, 128, 64), max_iter=300, use_mcr=False)
    mlp_r2_no_mcr, mlp_rmse_no_mcr = mlp_model_no_mcr.train()
    mlp_pred_v13_no_mcr, mlp_true_v13_no_mcr, mlp_angles_no_mcr = mlp_model_no_mcr.inverse_design()
    results['MLP (No MCR)'] = {
        'test_r2': mlp_r2_no_mcr,
        'test_rmse': mlp_rmse_no_mcr,
        'predicted_v13': mlp_pred_v13_no_mcr,
        'true_v13': mlp_true_v13_no_mcr,
        'prediction_error': abs(mlp_pred_v13_no_mcr - mlp_true_v13_no_mcr),
        'optimal_angles': mlp_angles_no_mcr.tolist(),
        'feature_type': 'Raw Angles'
    }
    
    # 5. LSTM模型（有MCR）
    print("\n" + "="*60)
    print("5. LONG SHORT-TERM MEMORY (LSTM) - With MCR (Small)")
    print("="*60)
    lstm_model_mcr = LSTM_Model(data_with_mcr, hidden_size=32, num_layers=1, num_epochs=100, use_mcr=True)
    lstm_r2_mcr, lstm_rmse_mcr = lstm_model_mcr.train()
    lstm_pred_v13_mcr, lstm_true_v13_mcr, lstm_angles_mcr = lstm_model_mcr.inverse_design()
    results['LSTM (MCR)'] = {
        'test_r2': lstm_r2_mcr,
        'test_rmse': lstm_rmse_mcr,
        'predicted_v13': lstm_pred_v13_mcr,
        'true_v13': lstm_true_v13_mcr,
        'prediction_error': abs(lstm_pred_v13_mcr - lstm_true_v13_mcr),
        'optimal_angles': lstm_angles_mcr.tolist(),
        'feature_type': 'MCR'
    }
    
    # 6. LSTM模型（无MCR）
    print("\n" + "="*60)
    print("6. LONG SHORT-TERM MEMORY (LSTM) - Without MCR (Small)")
    print("="*60)
    lstm_model_no_mcr = LSTM_Model(data_without_mcr, hidden_size=32, num_layers=1, num_epochs=100, use_mcr=False)
    lstm_r2_no_mcr, lstm_rmse_no_mcr = lstm_model_no_mcr.train()
    lstm_pred_v13_no_mcr, lstm_true_v13_no_mcr, lstm_angles_no_mcr = lstm_model_no_mcr.inverse_design()
    results['LSTM (No MCR)'] = {
        'test_r2': lstm_r2_no_mcr,
        'test_rmse': lstm_rmse_no_mcr,
        'predicted_v13': lstm_pred_v13_no_mcr,
        'true_v13': lstm_true_v13_no_mcr,
        'prediction_error': abs(lstm_pred_v13_no_mcr - lstm_true_v13_no_mcr),
        'optimal_angles': lstm_angles_no_mcr.tolist(),
        'feature_type': 'Raw Angles'
    }
    
    # 7. MCR-Former模型结果
    print("\n" + "="*60)
    print("7. MCR-Former (从先前结果获取)")
    print("="*60)
    mcr_angles = [-26.5, 24.5, 24.6, -23.3, -21.3 , 24.6,24.3, -26.0]
    mcr_pred_v13 = -0.3132
    mcr_true_v13 = get_v13(mcr_angles)
    mcr_r2 = 0.9862  # 替换为实际值
    mcr_rmse = 0.0087  # 替换为实际值
    
    print(f"MCR-Former Results:")
    print(f"  Predicted v13: {mcr_pred_v13:.6f}")
    print(f"  True v13: {mcr_true_v13:.6f}")
    print(f"  Prediction Error: {abs(mcr_pred_v13 - mcr_true_v13):.6f}")
    print(f"  Optimal Angles: {np.round(mcr_angles, 2)}")
    
    results['MCR-Former'] = {
        'test_r2': mcr_r2,
        'test_rmse': mcr_rmse,
        'predicted_v13': mcr_pred_v13,
        'true_v13': mcr_true_v13,
        'prediction_error': abs(mcr_pred_v13 - mcr_true_v13),
        'optimal_angles': mcr_angles,
        'feature_type': 'MCR'
    }
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f'model_comparison_all_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 打印整合的比较表格
    print("\n" + "="*120)
    print("COMPARISON TABLE - All Models")
    print("="*120)
    print(f"{'Model':<20} {'Feature':<12} {'Test R²':<10} {'Test RMSE':<12} {'Predicted v₁₃':<12} "
          f"{'True v₁₃':<12} {'Pred Error':<12} {'Best Angles'}")
    print("-"*120)
    
    for model_name, result in results.items():
        angles_str = ', '.join([f'{a:.1f}' for a in result['optimal_angles']])
        feature_type = result.get('feature_type', 'N/A')
        print(f"{model_name:<20} {feature_type:<12} {result['test_r2']:<10.4f} {result['test_rmse']:<12.6f} "
              f"{result['predicted_v13']:<12.6f} {result['true_v13']:<12.6f} "
              f"{result['prediction_error']:<12.6f} [{angles_str}]")
    
    # 分析结果
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # 找到最佳真实v13
    best_true_model = min(results.keys(), key=lambda x: results[x]['true_v13'])
    best_true_v13 = results[best_true_model]['true_v13']
    
    print(f"Best performing model (lowest true v13): {best_true_model}")
    print(f"Best true v13 value: {best_true_v13:.6f}")
    print(f"Best sequence: {results[best_true_model]['optimal_angles']}")
    
    # 对比理论最优
    theoretical_min = -0.3367
    print(f"\nComparison with theoretical optimum ({theoretical_min:.4f}):")
    for model_name, result in results.items():
        gap = abs(result['true_v13'] - theoretical_min)
        percent_gap = (gap / abs(theoretical_min)) * 100
        print(f"{model_name}: True v13 = {result['true_v13']:.6f}, "
              f"Gap = {gap:.6f} ({percent_gap:.2f}%)")
    
    # 模型预测误差分析
    print(f"\nPrediction Error Analysis:")
    for model_name, result in results.items():
        print(f"{model_name}: Predicted = {result['predicted_v13']:.6f}, "
              f"True = {result['true_v13']:.6f}, "
              f"Error = {result['prediction_error']:.6f}")
    
    # MCR vs No MCR 对比
    print("\n" + "="*80)
    print("MCR vs No MCR Comparison")
    print("="*80)
    
    mcr_models = [k for k in results.keys() if 'MCR' in k and 'No MCR' not in k and 'MCR-Former' not in k]
    no_mcr_models = [k for k in results.keys() if 'No MCR' in k]
    
    if mcr_models and no_mcr_models:
        avg_mcr_v13 = np.mean([results[m]['true_v13'] for m in mcr_models])
        avg_no_mcr_v13 = np.mean([results[m]['true_v13'] for m in no_mcr_models])
        avg_mcr_r2 = np.mean([results[m]['test_r2'] for m in mcr_models])
        avg_no_mcr_r2 = np.mean([results[m]['test_r2'] for m in no_mcr_models])
        
        print(f"Average true v13 with MCR: {avg_mcr_v13:.6f}")
        print(f"Average true v13 without MCR: {avg_no_mcr_v13:.6f}")
        print(f"Difference: {avg_no_mcr_v13 - avg_mcr_v13:.6f}")
        print(f"\nAverage R² with MCR: {avg_mcr_r2:.4f}")
        print(f"Average R² without MCR: {avg_no_mcr_r2:.4f}")
        print(f"Difference: {avg_mcr_r2 - avg_no_mcr_r2:.4f}")
    
    # 对称性分析
    def calculate_symmetry(angles):
        n = len(angles)
        if n < 2:
            return 0
        symmetry_score = 0
        for i in range(n//2):
            symmetry_score += abs(angles[i] + angles[n-1-i])  # 对于对称铺层，角度应该大小相等、符号相反
        return symmetry_score / (n//2)
    
    print(f"\nSymmetry Analysis:")
    for model_name, result in results.items():
        angles = result['optimal_angles']
        symmetry_score = calculate_symmetry(angles)
        print(f"{model_name}: Symmetry Score = {symmetry_score:.2f} (lower is better)")
    
    return results

if __name__ == "__main__":
    # 运行所有模型比较
    results = compare_all_models()
    
    print("\n" + "="*70)
    print("Comparison completed! Results saved to JSON file.")
    print("="*70)