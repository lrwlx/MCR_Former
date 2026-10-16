import torch
import numpy as np
import math
import os
import torch.nn as nn
from scipy.optimize import differential_evolution
import time
import json
import multiprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
from get_v13 import get_v13
warnings.filterwarnings('ignore')

# 从 MCR_Former.py 导入模型
from MCR_Former import MCR_Former

# 全局变量，用于并行计算
global_model = None
global_v13_scaler = None
global_num_layers = None
global_device = None
global_history_recorder = None

class HistoryRecorder:
    """记录每次优化迭代的历史结果"""
    def __init__(self):
        self.history = []
        self.restart_results = []
    
    def record_iteration(self, restart_id, iteration, best_value, best_angles, 
                        population_size, strategy):
        """记录单次迭代的结果"""
        self.history.append({
            'restart_id': restart_id,
            'iteration': iteration,
            'best_value': best_value,
            'best_angles': best_angles.copy(),
            'population_size': population_size,
            'strategy': strategy,
            'timestamp': time.time()
        })
    
    def record_restart_result(self, restart_id, best_value, best_angles, 
                             iterations, function_evaluations, elapsed_time,
                             strategy, real_v13=None, prediction_error=None):
        """记录每次重启的最终结果"""
        record = {
            'restart_id': restart_id,
            'best_value': best_value,
            'best_angles': best_angles.copy(),
            'iterations': iterations,
            'function_evaluations': function_evaluations,
            'elapsed_time_seconds': elapsed_time,
            'strategy': strategy,
            'timestamp': time.time()
        }
        
        # 添加真实v13值和预测误差（如果提供了）
        if real_v13 is not None:
            record['real_v13'] = real_v13
        if prediction_error is not None:
            record['prediction_error'] = prediction_error
            
        self.restart_results.append(record)
    
    def get_dataframe(self):
        """将所有历史记录转换为DataFrame"""
        if not self.history:
            return pd.DataFrame()
        
        # 创建历史记录的DataFrame
        history_df = pd.DataFrame(self.history)
        
        # 将角度列表展开为单独的列
        if len(self.history) > 0:
            num_layers = len(self.history[0]['best_angles'])
            for i in range(num_layers):
                history_df[f'angle_{i+1}'] = history_df['best_angles'].apply(lambda x: x[i])
        
        # 删除原始的best_angles列
        if 'best_angles' in history_df.columns:
            history_df = history_df.drop(columns=['best_angles'])
        
        return history_df
    
    def get_restart_results_df(self):
        """将重启结果转换为DataFrame"""
        if not self.restart_results:
            return pd.DataFrame()
        
        restart_df = pd.DataFrame(self.restart_results)
        
        # 将角度列表展开为单独的列
        if len(self.restart_results) > 0:
            num_layers = len(self.restart_results[0]['best_angles'])
            for i in range(num_layers):
                restart_df[f'angle_{i+1}'] = restart_df['best_angles'].apply(lambda x: x[i])
        
        # 删除原始的best_angles列
        if 'best_angles' in restart_df.columns:
            restart_df = restart_df.drop(columns=['best_angles'])
        
        # 按最佳值排序
        restart_df = restart_df.sort_values('best_value')
        restart_df = restart_df.reset_index(drop=True)
        restart_df.index.name = 'rank'
        restart_df = restart_df.reset_index()
        restart_df['rank'] = restart_df['rank'] + 1
        
        return restart_df

def init_worker(model_path, num_layers, device_type):
    """初始化工作进程，加载模型到指定设备"""
    global global_model, global_v13_scaler, global_num_layers, global_device, global_history_recorder
    
    # 设置设备
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    global_device = device
    global_num_layers = num_layers
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 重建模型 - 使用与训练时相同的MCR_Former模型
    config = checkpoint['config']
    model = MCR_Former(
        input_dim=2,  # 2D最小完备表示 (sin2θ, cos2θ)
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    global_model = model
    
    # 重建归一化器
    v13_scaler = StandardScaler()
    v13_scaler.mean_ = np.array([checkpoint['scaler_mean']])
    v13_scaler.scale_ = np.array([checkpoint['scaler_scale']])
    global_v13_scaler = v13_scaler
    
    # 初始化历史记录器
    global_history_recorder = HistoryRecorder()

def objective_function(angles):
    """
    差分进化的优化目标函数
    """
    global global_model, global_v13_scaler, global_num_layers, global_device
    
    if global_model is None:
        raise RuntimeError("模型未初始化！请先调用init_worker()")
    
    # 将角度转换为模型输入特征
    angles_tensor = torch.tensor(angles, dtype=torch.float32, device=global_device).view(1, -1)
    angles_rad = torch.deg2rad(angles_tensor)
    
    # 创建特征: [sin(2θ), cos(2θ)] - 最小完备表示
    features = torch.stack([
        torch.sin(2 * angles_rad),
        torch.cos(2 * angles_rad)
    ], dim=-1)
    
    # 创建长度张量
    lengths = torch.tensor([global_num_layers], dtype=torch.int64, device=global_device)
    
    # 模型预测 - 直接使用MCR_Former
    with torch.no_grad():
        global_model.eval()
        v13_pred = global_model(features, lengths)
    
    # 转换为原始尺度
    v13_original = global_v13_scaler.inverse_transform(
        v13_pred.cpu().numpy().reshape(-1, 1)
    )[0][0]
    
    return v13_original

def callback_func(xk, convergence):
    """
    差分进化的回调函数，记录每次迭代的最优值
    xk: 当前最优解
    convergence: 收敛值
    """
    global global_history_recorder, global_de_params
    if global_history_recorder and global_de_params:
        # 计算当前最优值
        current_best = objective_function(xk)
        
        # 获取当前重启ID、迭代次数和策略
        restart_id = global_de_params.get('restart_id', 0)
        iteration = global_de_params.get('iteration', 0)
        population_size = global_de_params.get('population_size', 0)
        strategy = global_de_params.get('strategy', 'unknown')
        
        # 记录当前迭代
        global_history_recorder.record_iteration(
            restart_id=restart_id,
            iteration=iteration,
            best_value=current_best,
            best_angles=xk.copy(),
            population_size=population_size,
            strategy=strategy
        )
        
        # 更新迭代次数
        global_de_params['iteration'] += 1

# 全局变量用于存储DE参数
global_de_params = {}

def calculate_real_v13_for_solutions(all_results, num_top_solutions=3):
    """
    为优化结果计算真实的v13值
    
    参数:
        all_results: 所有重启的结果列表
        num_top_solutions: 计算前N个最佳解的v13值
        
    返回:
        更新后的all_results（包含真实v13值和预测误差）
    """
    print(f"\n{'='*80}")
    print("计算真实v13值...")
    print('='*80)
    
    # 按最佳值排序
    sorted_results = sorted(all_results, key=lambda x: x['best_value'])
    
    # 只计算前N个最佳解的真实v13值
    top_solutions = sorted_results[:num_top_solutions]
    
    for i, res in enumerate(top_solutions):
        print(f"\n计算排名第 {i+1} 的解 (重启 #{res['restart_id']}):")
        print(f"  预测v13值: {res['best_value']:.8f}")
        print(f"  铺层角度: {np.round(res['best_angles'], 4)}")
        
        try:
            # 使用get_v13()计算真实v13值
            real_v13 = get_v13(res['best_angles'])
            res['real_v13'] = real_v13
            res['prediction_error'] = abs(real_v13 - res['best_value'])
            res['relative_error(%)'] = abs(real_v13 - res['best_value']) / abs(real_v13) * 100 if real_v13 != 0 else float('inf')
            
            print(f"  真实v13值: {real_v13:.8f}")
            print(f"  绝对误差: {res['prediction_error']:.8f}")
            print(f"  相对误差: {res['relative_error(%)']:.6f}%")
            
        except Exception as e:
            print(f"  计算真实v13值时出错: {e}")
            res['real_v13'] = None
            res['prediction_error'] = None
            res['relative_error(%)'] = None
    
    return all_results

def find_global_minimum(model_path, num_restarts=5, maxiter=200, popsize=25, 
                       calculate_real_v13=True, num_top_real_calculations=3):
    """
    使用差分进化算法寻找全局最小值
    
    参数:
        model_path: 训练好的模型文件路径
        num_restarts: 重启次数，用于避免局部最优
        maxiter: 最大迭代次数
        popsize: 种群大小参数（实际种群大小 = popsize * 维度）
        calculate_real_v13: 是否计算真实v13值
        num_top_real_calculations: 计算前N个最佳解的真实v13值
        
    返回:
        all_results: 所有重启的结果列表
        global_best_result: 全局最佳优化结果
        global_min_v13: 全局最小的v13值
        global_best_angles: 对应的全局最佳铺层角度
    """
    # 加载模型 (主进程)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    num_layers = config['num_layers_angle']
    
    # 重建归一化器 (主进程)
    v13_scaler = StandardScaler()
    v13_scaler.mean_ = np.array([checkpoint['scaler_mean']])
    v13_scaler.scale_ = np.array([checkpoint['scaler_scale']])
    
    # 角度范围: -90 到 90 度 (复合材料标准范围)
    bounds = [(-90, 90) for _ in range(num_layers)]
    
    # 初始化历史记录器
    history_recorder = HistoryRecorder()
    
    # 存储所有重启结果
    all_results = []
    global_min_v13 = float('inf')
    global_best_result = None
    global_best_angles = None
    
    print(f"开始全局优化搜索...")
    print(f"铺层层数: {num_layers}")
    print(f"角度范围: -90° 到 90°")
    print(f"重启次数: {num_restarts}")
    print(f"每次最大迭代: {maxiter}")
    print(f"基础种群大小: {popsize}")
    print(f"实际种群大小: {popsize * num_layers}")
    print(f"计算真实v13值: {calculate_real_v13}")
    print(f"计算前N个最佳解的真实v13值: {num_top_real_calculations}")
    print("-" * 80)
    
    # DE策略列表
    strategies = ['rand1bin', 'rand2bin', 'best1bin', 'best2bin']
    
    # 创建并行池 (保留一个核心给系统)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # 进行多次重启以避免局部最优
    for restart in range(num_restarts):
        print(f"\n{'='*60}")
        print(f"重启 #{restart+1}/{num_restarts}")
        print(f"{'='*60}")
        
        # 随机选择DE策略
        strategy = np.random.choice(strategies)
        print(f"使用的策略: {strategy}")
        
        # 为每个重启创建独立的进程池
        with multiprocessing.Pool(processes=num_cores, initializer=init_worker,
                                 initargs=(model_path, num_layers, str(device))) as pool:
            
            # 设置全局DE参数，用于回调函数
            global global_de_params
            global_de_params = {
                'restart_id': restart + 1,
                'iteration': 0,
                'population_size': popsize * num_layers,
                'strategy': strategy
            }
            
            start_time = time.time()
            
            # 运行差分进化
            result = differential_evolution(
                func=objective_function,
                bounds=bounds,
                strategy=strategy,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-6,
                polish=True,
                updating='deferred',
                workers=pool.map,
                callback=callback_func,  # 添加回调函数记录历史
                disp=True
            )
            
            elapsed_time = time.time() - start_time
            
            # 记录本次重启的结果
            history_recorder.record_restart_result(
                restart_id=restart + 1,
                best_value=result.fun,
                best_angles=result.x,
                iterations=result.nit,
                function_evaluations=result.nfev,
                elapsed_time=elapsed_time,
                strategy=strategy
            )
            
            # 保存本次重启结果
            restart_result = {
                'restart_id': restart + 1,
                'result': result,
                'best_value': result.fun,
                'best_angles': result.x,
                'elapsed_time': elapsed_time,
                'strategy': strategy
            }
            all_results.append(restart_result)
            
            print(f"\n重启 #{restart+1} 完成!")
            print(f"  最终 v13 值: {result.fun:.8f}")
            print(f"  迭代次数: {result.nit}")
            print(f"  函数评估次数: {result.nfev}")
            print(f"  耗时: {elapsed_time:.2f}秒")
            print(f"  最佳铺层序列: {np.round(result.x, 4)}")
            
            # 更新全局最佳结果
            if result.fun < global_min_v13:
                global_min_v13 = result.fun
                global_best_result = result
                global_best_angles = result.x
                print(f"  --> 发现新的全局最优解!")
    
    print("\n" + "="*80)
    print("全局优化完成!")
    print("="*80)
    
    # 如果启用了真实v13计算，计算前N个最佳解的真实v13值
    if calculate_real_v13:
        all_results = calculate_real_v13_for_solutions(
            all_results, 
            num_top_solutions=num_top_real_calculations
        )
    
    # 打印所有重启结果
    print("\n所有重启结果汇总:")
    print("-"*80)
    
    # 按最佳值排序
    sorted_results = sorted(all_results, key=lambda x: x['best_value'])
    
    for i, res in enumerate(sorted_results):
        print(f"\n排名 {i+1}: 重启 #{res['restart_id']}")
        print(f"  v13值: {res['best_value']:.8f}")
        
        # 如果计算了真实v13值，显示相关信息
        if 'real_v13' in res and res['real_v13'] is not None:
            print(f"  真实v13值: {res['real_v13']:.8f}")
            print(f"  预测误差: {res.get('prediction_error', 'N/A'):.8f}")
            print(f"  相对误差: {res.get('relative_error(%)', 'N/A'):.6f}%")
            
        print(f"  耗时: {res['elapsed_time']:.2f}秒")
        print(f"  策略: {res['strategy']}")
        print(f"  铺层角度: {np.round(res['best_angles'], 4)}")
    
    print("\n" + "="*80)
    
    # 确定全局最佳结果（考虑真实v13值）
    if calculate_real_v13 and 'real_v13' in sorted_results[0]:
        # 如果有真实v13值，使用真实值排序
        global_min_v13 = sorted_results[0]['real_v13']
        global_best_angles = sorted_results[0]['best_angles']
        global_best_result = sorted_results[0]['result']
        best_restart_id = sorted_results[0]['restart_id']
        print(f"全局最佳结果 (基于真实v13值, 来自重启 #{best_restart_id}):")
        print(f"  最小真实 v13 值: {global_min_v13:.8f}")
        print(f"  模型预测 v13 值: {sorted_results[0]['best_value']:.8f}")
        print(f"  预测误差: {sorted_results[0].get('prediction_error', 'N/A'):.8f}")
    else:
        # 否则使用模型预测值
        best_restart_id = sorted_results[0]['restart_id']
        print(f"全局最佳结果 (基于模型预测, 来自重启 #{best_restart_id}):")
        print(f"  最小预测 v13 值: {global_min_v13:.8f}")
    
    print(f"  最佳铺层序列: {np.round(global_best_angles, 4)}")
    print("="*80)
    
    # 验证对称性（使用真实v13值如果可用）
    print("\n验证全局最优解的对称性...")
    
    # 重新初始化模型 (主进程) - 使用MCR_Former
    model = MCR_Former(
        input_dim=2,  # 2D最小完备表示
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 原始序列预测
    angles_tensor = torch.tensor(global_best_angles, dtype=torch.float32, device=device).view(1, -1)
    angles_rad = torch.deg2rad(angles_tensor)
    features = torch.stack([
        torch.sin(2 * angles_rad),
        torch.cos(2 * angles_rad)
    ], dim=-1)
    lengths = torch.tensor([num_layers], dtype=torch.int64, device=device)
    
    with torch.no_grad():
        forward_pred = model(features, lengths)
        forward_v13 = v13_scaler.inverse_transform(
            forward_pred.cpu().numpy().reshape(-1, 1)
        )[0][0]
    
    # 反转序列预测
    reversed_features = torch.flip(features, dims=[1])
    with torch.no_grad():
        reversed_pred = model(reversed_features, lengths)
        reversed_v13 = v13_scaler.inverse_transform(
            reversed_pred.cpu().numpy().reshape(-1, 1)
        )[0][0]
    
    symmetry_diff = abs(forward_v13 - reversed_v13)
    print(f"原始序列预测 v13: {forward_v13:.8f}")
    print(f"反转序列预测 v13: {reversed_v13:.8f}")
    print(f"预测对称性差异: {symmetry_diff:.8e}")
    
    # 计算真实对称性（如果启用了真实v13计算）
    if calculate_real_v13:
        try:
            print("\n计算真实对称性...")
            real_forward_v13 = get_v13(global_best_angles)
            real_reversed_v13 = get_v13(np.flip(global_best_angles))
            real_symmetry_diff = abs(real_forward_v13 - real_reversed_v13)
            
            print(f"原始序列真实 v13: {real_forward_v13:.8f}")
            print(f"反转序列真实 v13: {real_reversed_v13:.8f}")
            print(f"真实对称性差异: {real_symmetry_diff:.8e}")
            
        except Exception as e:
            print(f"计算真实对称性时出错: {e}")
    
    # 保存结果到Excel
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    excel_filename = f'global_optimization_results_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 保存所有迭代历史
        history_df = history_recorder.get_dataframe()
        if not history_df.empty:
            history_df.to_excel(writer, sheet_name='All_Iterations', index=False)
            print(f"\n已保存 {len(history_df)} 条迭代记录到 '{excel_filename}' 的 'All_Iterations' sheet")
        
        # 保存重启结果汇总
        restart_df = history_recorder.get_restart_results_df()
        if not restart_df.empty:
            restart_df.to_excel(writer, sheet_name='Restart_Results', index=False)
            print(f"已保存 {len(restart_df)} 条重启结果到 'Restart_Results' sheet")
        
        # 保存全局最优结果
        global_result_data = {
            'global_best_value': global_min_v13,
            'global_best_restart_id': best_restart_id,
            'model_prediction_v13': float(sorted_results[0]['best_value']),
            'forward_v13': forward_v13,
            'reversed_v13': reversed_v13,
            'symmetry_diff': symmetry_diff,
            'total_restarts': num_restarts,
            'total_function_evaluations': sum(r['result'].nfev for r in all_results),
            'total_time_seconds': sum(r['elapsed_time'] for r in all_results)
        }
        
        # 添加真实v13值信息（如果可用）
        if calculate_real_v13 and 'real_v13' in sorted_results[0]:
            global_result_data['real_v13'] = float(sorted_results[0]['real_v13'])
            global_result_data['prediction_error'] = float(sorted_results[0].get('prediction_error', 0))
            global_result_data['relative_error(%)'] = float(sorted_results[0].get('relative_error(%)', 0))
        
        global_result_df = pd.DataFrame([global_result_data])
        
        # 添加角度列
        for i in range(num_layers):
            global_result_df[f'angle_{i+1}'] = global_best_angles[i]
        
        global_result_df.to_excel(writer, sheet_name='Global_Best', index=False)
        print(f"已保存全局最优结果到 'Global_Best' sheet")
    
    # 保存详细结果到JSON
    json_results = {
        'global_best': {
            'min_v13': float(global_min_v13),
            'optimal_angles': global_best_angles.tolist(),
            'restart_id': best_restart_id,
            'strategy': sorted_results[0]['strategy'],
            'model_prediction_v13': float(sorted_results[0]['best_value'])
        },
        'all_restarts': [],
        'num_layers': num_layers,
        'num_restarts': num_restarts,
        'symmetry_validation': {
            'forward_v13': float(forward_v13),
            'reversed_v13': float(reversed_v13),
            'symmetry_diff': float(symmetry_diff),
            'valid': bool(symmetry_diff < 1e-6)
        },
        'optimization_time': time.time(),
        'angle_range': [-90, 90],
        'excel_filename': excel_filename,
        'calculate_real_v13': calculate_real_v13
    }
    
    # 如果计算了真实v13值，添加到JSON
    if calculate_real_v13 and 'real_v13' in sorted_results[0]:
        json_results['global_best']['real_v13'] = float(sorted_results[0]['real_v13'])
        json_results['global_best']['prediction_error'] = float(sorted_results[0].get('prediction_error', 0))
        json_results['global_best']['relative_error(%)'] = float(sorted_results[0].get('relative_error(%)', 0))
    
    # 添加所有重启的详细信息
    for res in all_results:
        restart_info = {
            'restart_id': res['restart_id'],
            'best_v13': float(res['best_value']),
            'angles': res['best_angles'].tolist(),
            'iterations': res['result'].nit,
            'function_evaluations': res['result'].nfev,
            'elapsed_time': res['elapsed_time'],
            'strategy': res['strategy']
        }
        
        # 如果计算了真实v13值，添加到重启信息
        if 'real_v13' in res and res['real_v13'] is not None:
            restart_info['real_v13'] = float(res['real_v13'])
            restart_info['prediction_error'] = float(res.get('prediction_error', 0))
            restart_info['relative_error(%)'] = float(res.get('relative_error(%)', 0))
        
        json_results['all_restarts'].append(restart_info)
    
    json_filename = f'global_optimization_details_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"\n详细结果已保存到 '{json_filename}'")
    print(f"所有结果已保存到Excel文件: '{excel_filename}'")

    return all_results, global_best_result, global_min_v13, global_best_angles

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Windows系统需要设置多进程启动方法
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn', force=True)
    
    # 执行全局优化
    model_path = 'mcr_former_best_model_with_aug.pth'  # 使用训练时保存的模型文件名
    
    try:
        all_results, global_best, min_v13, optimal_angles = find_global_minimum(
            model_path=model_path,
            num_restarts=2,    # 重启次数
            maxiter=50,       # 每次最大迭代次数
            popsize=50,        # 种群大小参数
            calculate_real_v13=True,  # 启用真实v13计算
            num_top_real_calculations=3  # 计算前3个最佳解的真实v13值
        )
        
        print("\n" + "="*80)
        print("优化完成!")
        print("="*80)
        
    except Exception as e:
        print(f"\n优化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()