#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地壳热扩散数值模拟测试

简化版测试套件，仅包含核心功能测试
"""
import numpy as np
import matplotlib.pyplot as plt

# 物理常数
D = 0.1       # 热扩散率 (m^2/day)
A = 10.0      # 年平均地表温度 (°C)
B = 12.0      # 地表温度振幅 (°C)
TAU = 365.0   # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0 # 初始温度 (°C)
DEPTH_MAX = 20.0 # 最大深度 (m)


def solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, N=366, years=10):
    """
    显式有限差分法求解地壳热扩散方程
    
    参数:
        h (float): 空间步长 (m)
        a (float): 时间步长比例因子
        M (int): 深度方向网格点数
        N (int): 时间步数
        years (int): 总模拟年数
    
    返回:
        tuple: (depth_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)
            - temperature_matrix (ndarray): 温度矩阵 [depth, time]
    """
    # 计算时间步长和稳定性参数
    dt = a**2 / D  # 时间步长
    r = D * dt / (h**2)  # 稳定性参数
    
    print(f"空间步长 h = {h:.2f} m, 时间步长 dt = {dt:.2f} days, 稳定性参数 r = {r:.4f}")
    
    # 检查稳定性条件
    if r > 0.5:
        print(f"警告: 稳定性参数 r = {r:.4f} > 0.5，显式差分格式可能不稳定")
    
    # 初始化温度矩阵 [depth, time]
    T = np.zeros((M, N)) + T_INITIAL
    depth = np.linspace(0, DEPTH_MAX, M)  # 深度数组
    
    # 底部边界条件
    T[-1, :] = T_BOTTOM
    
    # 多年循环，只保留最后一年的结果
    for _ in range(years):
        # 时间步进循环
        for j in range(N-1):
            # 地表边界条件 (随时间变化)
            day_of_year = j % 365  # 计算当年的天数
            T[0, j] = A + B * np.sin(2 * np.pi * day_of_year / TAU)
            
            # 显式差分格式更新内部节点
            for i in range(1, M-1):
                T[i, j+1] = T[i, j] + r * (T[i+1, j] + T[i-1, j] - 2*T[i, j])
    
    return depth, T


def plot_seasonal_profiles(depth, temperature, seasons=[90, 180, 270, 365]):
    """
    绘制季节性温度轮廓
    
    参数:
        depth (ndarray): 深度数组
        temperature (ndarray): 温度矩阵 [depth, time]
        seasons (list): 季节时间点 (days)
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制各季节的温度轮廓
    for day in seasons:
        if day < temperature.shape[1]:
            plt.plot(depth, temperature[:, day], label=f'Day {day}')
    
    plt.xlabel('Depth (m)')
    plt.ylabel('Temperature (°C)')
    plt.title('Seasonal Temperature Profiles')
    plt.grid(True)
    plt.legend()
    plt.ylim(min(temperature.min(), -50), max(temperature.max(), 50))
    
    # 保存图像而不是显示，便于GitHub Actions等环境使用
    plt.savefig('seasonal_temperature_profiles.png')
    plt.close()


if __name__ == "__main__":
    # 运行模拟 (使用默认参数)
    depth, T = solve_earth_crust_diffusion()
    
    # 绘制季节性温度轮廓
    plot_seasonal_profiles(depth, T)
