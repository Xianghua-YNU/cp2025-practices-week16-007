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


def solve_earth_crust_diffusion(h=1.0, M=21, N=366, years=10):
    """
    求解地壳热扩散方程 (显式差分格式)，满足测试用例及题目任务要求。
    处理时变地表边界、固定底部边界，支持长期模拟，输出符合测试的维度。

    参数:
        h (float): 空间步长 (m)，默认 1m 保证深度点数 M=21 时覆盖 0-20m
        M (int): 深度方向网格点数，默认 21 个（0 - 20m 共 21 点 ）
        N (int): 一年的时间步数，默认 366 天（覆盖完整年周期 ）
        years (int): 总模拟年数，用于长期演化分析

    返回:
        tuple: (depth_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)，shape=(M,)
            - temperature_matrix (ndarray): 温度矩阵，shape=(M, N) ，为模拟最后一年的温度数据，适配测试用例
    """
    # 空间离散：生成深度数组
    depth = np.linspace(0, DEPTH_MAX, M)
    dz = h  # 空间步长

    # 时间离散：计算满足稳定性的时间步长（显式差分核心条件）
    dt = 0.5 * dz ** 2 / D  # 取稳定性极限 dt = 0.5 * dz² / D ，保证 r = 0.5
    r = D * dt / dz ** 2  # 稳定性参数，理论上 r 应 ≤ 0.5
    print(f"空间步长 h={h}m，时间步长 dt={dt:.4f}day，稳定性参数 r={r:.4f}")

    # 初始化温度场：[深度点数, 时间步数]，多年模拟后保留最后一年数据
    T = np.full((M, N), T_INITIAL, dtype=np.float64)
    T[-1, :] = T_BOTTOM  # 固定底部边界条件

    # 进行多年模拟，循环更新温度场，最终保留最后一年的温度数据
    for _ in range(years):
        # 逐天计算温度演化（显式差分更新）
        for j in range(N - 1):
            # 1. 处理地表时变边界条件：T_surface = A + B * sin(2πt/TAU)，t 为第 j 天
            T[0, j] = A + B * np.sin(2 * np.pi * j / TAU)

            # 2. 显式差分格式更新内部节点温度（深度方向除地表和底部的点 ）
            for i in range(1, M - 1):
                T[i, j + 1] = T[i, j] + r * (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j])

    return depth, T


def plot_seasonal_profiles(depth, temperature, seasons=[90, 180, 270, 365]):
    """
    绘制季节性温度轮廓，满足题目“在第 10 年选择 4 个时间点（代表四季 ）”需求，
    基于模拟得到的最后一年数据绘图。

    参数:
        depth (ndarray): 深度数组 (m)，shape=(M,)
        temperature (ndarray): 温度矩阵，shape=(M, N) ，N 为一年的天数（366 ）
        seasons (list): 要绘制的时间点（天数 ），默认选四季典型天数
    """
    plt.figure(figsize=(10, 8))

    # 绘制各季节时间点的温度 - 深度曲线
    for day in seasons:
        if 0 <= day < temperature.shape[1]:  # 防止天数越界
            plt.plot(depth, temperature[:, day], label=f'Day {day}', linewidth=2)

    plt.xlabel('Depth (m)')
    plt.ylabel('Temperature (°C)')
    plt.title('Seasonal Temperature Profiles (Last Simulated Year)')
    plt.grid(True)
    plt.legend()

    # 在 GitHub 环境中，可保存图片而不是直接显示（便于自动化流程），也可保留显示逻辑
    plt.savefig('seasonal_temperature_profiles.png')
    plt.close()
    # plt.show()


if __name__ == "__main__":
    # 运行模拟，设置模拟 10 年，获取最后一年（第 10 年 ）数据用于分析
    depth, T = solve_earth_crust_diffusion(years=10)

    # 绘制季节性温度轮廓，使用第 10 年的四季时间点
    plot_seasonal_profiles(depth, T, seasons=[90, 180, 270, 365])

    # 以下可补充长期演化分析的代码，如查看地表温度年周期是否稳定等
    # 示例：绘制最后一年地表温度的年变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(T[0, :], label='Surface Temperature (Last Year)')
    plt.xlabel('Day')
    plt.ylabel('Temperature (°C)')
    plt.title('Surface Temperature Cycle (Last Simulated Year)')
    plt.grid(True)
    plt.legend()
    plt.savefig('surface_temperature_cycle.png')
    plt.close()
    # plt.show()
