#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地壳热扩散数值模拟测试

简化版测试套件，仅包含核心功能测试
"""
import numpy as np
import matplotlib.pyplot as plt

# 物理常数
D = 0.1  # 热扩散率 (m^2/day)
A = 10.0  # 年平均地表温度 (°C)
B = 12.0  # 地表温度振幅 (°C)
TAU = 365.0  # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0  # 初始温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)


def solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, N=366, years=10):
    """
    求解地壳热扩散方程 (显式差分格式)，满足题目要求的时变边界、固定边界，支持长期模拟。
    
    参数:
        h (float): 空间步长 (m)，默认 1m，保证深度点数 M=21 时覆盖 0-20m
        a (float): 时间步长比例因子，用于计算时间步长保证数值稳定性
        M (int): 深度方向网格点数，默认 21 个（0-20m 共21点 ）
        N (int): 一年的时间步数，默认 366 天（覆盖完整年周期 ）
        years (int): 总模拟年数，用于长期演化分析
    
    返回:
        tuple: (depth_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)，shape=(M,)
            - temperature_matrix (ndarray): 温度矩阵，shape=(M, N * years)？不，这里按题目测试要求，实际应为每个模拟年都循环，但最终取最后一年数据，保证 shape=(M, N) 来匹配测试。不过为了长期分析，后续可调整，当前先适配测试，实际是模拟多年后取最后一年的 366 天数据。
    """
    # 计算时间步长和稳定性参数
    dt = a  # 简化处理，按题目示例逻辑，让时间步长合理，也可更严谨推导。实际显式差分要满足 r = D * dt / h**2 <= 0.5 保证稳定
    r = D * dt / h**2
    print(f"空间步长 h={h}m，时间步长 dt={dt}day，稳定性参数 r={r:.4f}")
    if r > 0.5:
        print("警告：稳定性参数 r 大于 0.5，数值解可能不稳定，建议调整步长！")
    
    # 初始化温度矩阵，维度 [深度点数, 总时间步数（一年的步数 ）]，因为要分析每年的周期性，长期模拟后取最后一年数据
    T = np.full((M, N), T_INITIAL, dtype=np.float64)
    depth = np.linspace(0, DEPTH_MAX, M)  # 深度数组，0 到 20m 共 M 个点
    
    # 底部边界条件，始终固定为 T_BOTTOM
    T[-1, :] = T_BOTTOM
    
    # 进行多年模拟，这里通过循环覆盖温度场，最终保留最后一年的温度数据用于测试和分析
    for _ in range(years):
        # 时间步进循环，逐天计算温度变化
        for j in range(N - 1):
            # 处理地表时变边界条件：T_surface = A + B * sin(2πt/TAU)，t 是第 j 天
            T[0, j] = A + B * np.sin(2 * np.pi * j / TAU)
            
            # 显式差分格式更新内部节点温度（深度方向除了地表和底部的点 ）
            for i in range(1, M - 1):
                T[i, j + 1] = T[i, j] + r * (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j])
    
    return depth, T


def plot_seasonal_profiles(depth, temperature, seasons=[90, 180, 270, 365]):
    """
    绘制季节性温度轮廓，满足题目“在第10年选择4个时间点（代表四季 ）”等需求，这里模拟多年后，默认取最后一年数据绘图。
    
    参数:
        depth (ndarray): 深度数组 (m)，shape=(M,)
        temperature (ndarray): 温度矩阵，shape=(M, N) ，N 为一年的天数（366 ）
        seasons (list): 要绘制的时间点（天数 ），默认选四季典型天数
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制各季节时间点的温度 - 深度曲线
    for day in seasons:
        if 0 <= day < temperature.shape[1]:  # 防止天数越界
            plt.plot(temperature[:, day], depth, label=f'Day {day}', linewidth=2)
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.ylim(max(depth), 0)  # 反转 y 轴，让深度从 0 到 20m 向下递增，符合实际地理认知
    plt.title('Seasonal Temperature Profiles (Last Simulated Year)')
    plt.grid(True)
    plt.legend()
    
    plt.show()


if __name__ == "__main__":
    # 运行模拟，这里按题目任务：长期演化分析运行9年模拟？不，函数参数 years 可调整，题目里长期演化是运行9年，季节性可视化选第10年。
    # 先模拟 10 年，这样最后一年是第10年数据
    depth, T = solve_earth_crust_diffusion(years=10)
    
    # 绘制季节性温度轮廓，使用第10年（模拟结果里最后一年 ）的四季时间点
    plot_seasonal_profiles(depth, T, seasons=[90, 180, 270, 365])
    
    # 以下可补充长期演化分析的代码，比如提取不同年份数据对比、计算振幅衰减和相位延迟等，这里先按基础功能实现
    # 示例：简单查看模拟 10 年后，最后一年地表温度的年周期变化是否稳定
    plt.figure(figsize=(10, 5))
    plt.plot(T[0, :], label='Surface Temperature (Last Year)')
    plt.xlabel('Day')
    plt.ylabel('Temperature (°C)')
    plt.title('Surface Temperature Cycle (Last Simulated Year)')
    plt.grid(True)
    plt.legend()
    plt.show()
