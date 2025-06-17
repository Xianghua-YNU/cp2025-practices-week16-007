"""学生模板：量子隧穿效应
文件：quantum_tunneling_student.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体（或你系统有的中文字体）
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

class QuantumTunnelingSolver:
    """量子隧穿求解器类
    
    该类实现了一维含时薛定谔方程的数值求解，用于模拟量子粒子的隧穿效应。
    使用变形的Crank-Nicolson方法进行时间演化，确保数值稳定性和概率守恒。
    """
    
    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        """初始化量子隧穿求解器
        
        参数:
            Nx (int): 空间网格点数，默认220
            Nt (int): 时间步数，默认300
            x0 (float): 初始波包中心位置，默认40
            k0 (float): 初始波包动量(波数)，默认0.5
            d (float): 初始波包宽度参数，默认10
            barrier_width (int): 势垒宽度，默认3
            barrier_height (float): 势垒高度，默认1.0
        """
        # TODO: 初始化所有参数
        # 提示：需要设置空间网格、势垒参数，并初始化波函数和系数矩阵
        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x0
        self.k0 = k0
        self.d = d
        self.barrier_width = int(barrier_width)  # 确保是整数
        self.barrier_height = barrier_height
        
        # TODO: 创建空间网格
        self.x = np.arange(self.Nx)

        # TODO: 设置势垒
        self.V = self.setup_potential()  # 调用 setup_potential() 方法

        # TODO: 初始化波函数矩阵和系数矩阵
        self.C = np.zeros((self.Nx, self.Nt), dtype=np.complex128)  # 复数矩阵，形状为 (Nx, Nt)
        self.B = np.zeros((self.Nx, self.Nt), dtype=np.complex128)  # 复数矩阵，形状为 (Nx, Nt)


    def wavefun(self, x):
        """高斯波包函数
        
        参数:
            x (np.ndarray): 空间坐标数组
            
        返回:
            np.ndarray: 初始波函数值
            
        数学公式:
            ψ(x,0) = exp(ik₀x) * exp(-(x-x₀)²ln10(2)/d²)
            
        物理意义:
            描述一个在x₀位置、具有动量k₀、宽度为d的高斯波包
        """
        # TODO: 实现高斯波包函数
        # 提示：包含动量项 exp(ik₀x) 和高斯包络 exp(-(x-x₀)²ln10(2)/d²)
        return np.exp(1j * self.k0 * x) * np.exp(-(x - self.x0) ** 2 * np.log10(2) / (self.d ** 2))

    def setup_potential(self):
        """设置势垒函数
        
        返回:
            np.ndarray: 势垒数组
            
        说明:
            在空间网格中间位置创建矩形势垒
            势垒位置：从 Nx//2 到 Nx//2+barrier_width
            势垒高度：barrier_height
        """
        # TODO: 创建势垒数组
        # 提示：
        # 1. 初始化全零数组
        # 2. 在中间位置设置势垒高度
        # 3. 注意barrier_width必须是整数
        V = np.zeros(self.Nx)   # 势垒数组初始化为全零
        mid = self.Nx // 2      # 势垒中心位置
        V[mid:mid + self.barrier_width] = self.barrier_height  # 设置势垒高度
        return V

    def build_coefficient_matrix(self):
        """构建变形的Crank-Nicolson格式的系数矩阵
        
        返回:
            np.ndarray: 系数矩阵A
            
        数学原理:
            对于dt=1, dx=1的情况，哈密顿矩阵的对角元素为: -2+2j-V
            非对角元素为1（表示动能项的有限差分）
            
        矩阵结构:
            三对角矩阵，主对角线为 -2+2j-V[i]，上下对角线为1
        """
        # TODO: 构建系数矩阵
        # 提示：
        # 1. 使用 np.diag() 创建三对角矩阵
        # 2. 主对角线：-2+2j-self.V
        # 3. 上对角线和下对角线：全1数组
        main_diag = -2 + 2j - self.V  # 主对角线
        off_diag = np.ones(self.Nx - 1)  # 上下对角线为1
        A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    def solve_schrodinger(self):
        """求解一维含时薛定谔方程
        
        使用Crank-Nicolson方法进行时间演化
        
        返回:
            tuple: (x, V, B, C) - 空间网格, 势垒, 波函数矩阵, chi矩阵
            
        数值方法:
            Crank-Nicolson隐式格式，具有二阶精度和无条件稳定性
            时间演化公式：C[:,t+1] = 4j * solve(A, B[:,t])
                         B[:,t+1] = C[:,t+1] - B[:,t]
        """
        # TODO: 实现薛定谔方程求解
        # 提示：
        # 1. 构建系数矩阵A
        # 2. 设置初始波函数 B[:,0] = wavefun(x)
        # 3. 对初始波函数进行归一化
        # 4. 时间循环：使用线性方程组求解进行时间演化
        A = self.build_coefficient_matrix()
        self.B[:, 0] = self.wavefun(self.x)  # 设置初始波函数
        self.B[:, 0] /= np.sqrt(np.trapz(np.abs(self.B[:, 0]) ** 2, self.x)) # 归一化初始波函数
        for t in range(self.Nt - 1):
            # 使用线性方程组求解
            self.C[:, t + 1] = 4j * np.linalg.solve(A, self.B[:, t])
            self.B[:, t + 1] = self.C[:, t + 1] - self.B[:, t]
        return (self.x, self.V, self.B, self.C)

    def calculate_coefficients(self):
        """计算透射和反射系数
        
        返回:
            tuple: (T, R) - 透射系数和反射系数
            
        物理意义:
            透射系数T：粒子穿过势垒的概率
            反射系数R：粒子被势垒反射的概率
            应满足：T + R ≈ 1（概率守恒）
            
        计算方法:
            T = ∫|ψ(x>barrier)|²dx / ∫|ψ(x)|²dx
            R = ∫|ψ(x<barrier)|²dx / ∫|ψ(x)|²dx
        """
        # TODO: 计算透射和反射系数
        # 提示：
        # 1. 确定势垒位置
        # 2. 计算透射区域的概率（势垒右侧）
        # 3. 计算反射区域的概率（势垒左侧）
        # 4. 归一化处理
        barrier_start = self.Nx // 2
        barrier_end = barrier_start + self.barrier_width
        # 计算透射区域的概率
        T = np.trapz(np.abs(self.B[barrier_end:, -1]) ** 2, self.x[barrier_end:]) / np.trapz(np.abs(self.B[:, -1]) ** 2, self.x)
        # 计算反射区域的概率
        R = np.trapz(np.abs(self.B[:barrier_start, -1]) ** 2, self.x[:barrier_start]) / np.trapz(np.abs(self.B[:, -1]) ** 2, self.x)
        # 归一化处理
        T = T / (T + R)
        R = R / (T + R)
        return (T, R)

    def plot_evolution(self, time_indices=None):
        """绘制波函数演化图
        
        参数:
            time_indices (list): 要绘制的时间索引列表，默认为[0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
            
        功能:
            在多个子图中显示不同时刻的波函数概率密度和势垒
        """
        # TODO: 实现波函数演化绘图
        # 提示：
        # 1. 设置默认时间索引
        # 2. 创建子图布局
        # 3. 绘制概率密度 |ψ|²
        # 4. 绘制势垒
        # 5. 添加标题和标签
        if time_indices is None: # 默认时间索引
            time_indices = [0, self.Nt // 4, self.Nt // 2, 3 * self.Nt // 4, self.Nt - 1]
        num_plots = len(time_indices) # 创建子图数量
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 3), sharey=True) # 创建子图
        for i, t in enumerate(time_indices): # 绘制每个时间步的波函数
            axes[i].plot(self.x, np.abs(self.B[:, t]) ** 2, label=fr'$|\psi(x,{t})|^2$') # 绘制波函数概率密度
            axes[i].plot(self.x, self.V, label='势垒', color='red') # 绘制势垒
            axes[i].set_title(f'Time step {t}') # 设置标题
            axes[i].set_xlabel('Position (x)') # 设置x轴标签
            axes[i].set_ylabel('Probability Density') # 设置y轴标签
            axes[i].legend() # 添加图例
        plt.tight_layout()
        plt.show()  # 显示图形

    def create_animation(self, interval=20):
        """创建波包演化动画
        
        参数:
            interval (int): 动画帧间隔(毫秒)，默认20
            
        返回:
            matplotlib.animation.FuncAnimation: 动画对象
            
        功能:
            实时显示波包在势垒附近的演化过程
        """
        # TODO: 创建动画
        # 提示：
        # 1. 设置图形和坐标轴
        # 2. 创建线条对象
        # 3. 定义动画更新函数
        # 4. 使用 FuncAnimation 创建动画
        fig, ax = plt.subplots(figsize=(8, 4))  # 创建图形和坐标轴
        line, = ax.plot([], [], lw=2)  # 创建线条对象
        ax.set_xlim(0, self.Nx - 1)  # 设置x轴范围
        ax.set_ylim(0, 1.2 * np.max(np.abs(self.B) ** 2))  # 设置y轴范围
        ax.set_xlabel('Position (x)')  # 设置x轴标签
        ax.set_ylabel('Probability Density')  # 设置y轴标签
        ax.set_title('Quantum Tunneling Evolution')  # 设置标题
        ax.grid()  # 添加网格
        # 初始化函数
        def init():
            """初始化动画"""
            line.set_data([], [])
            return line,
        # 更新函数
        def update(frame):
            """更新动画帧"""
            line.set_data(self.x, np.abs(self.B[:, frame]) ** 2)
            ax.set_title(f'Quantum Tunneling Evolution at Time Step {frame}')  # 更新标题
            return line,
        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=self.Nt, init_func=init, blit=True, interval=interval)
        return ani

    def verify_probability_conservation(self):
        """验证概率守恒
        
        返回:
            np.ndarray: 每个时间步的总概率
            
        物理原理:
            量子力学中概率必须守恒：∫|ψ(x,t)|²dx = 常数
            数值计算中应该保持在1附近
        """
        # TODO: 验证概率守恒
        # 提示：
        # 1. 计算每个时间步的总概率
        # 2. 考虑空间步长dx的影响
        # 3. 返回概率数组用于分析
        probabilities = np.zeros(self.Nt) # 初始化概率数组
        dx = self.x[1] - self.x[0] # 计算空间步长
        for t in range(self.Nt): # 计算每个时间步的总概率
            # 使用数值积分计算总概率
            probabilities[t] = np.trapz(np.abs(self.B[:, t]) ** 2, self.x) * dx
        return probabilities

    def demonstrate(self):
        """演示量子隧穿效应
        
        功能:
            1. 求解薛定谔方程
            2. 计算并显示透射和反射系数
            3. 绘制波函数演化图
            4. 验证概率守恒
            5. 创建并显示动画
            
        返回:
            animation对象
        """
        # TODO: 实现完整的演示流程
        # 提示：
        # 1. 打印开始信息
        # 2. 调用solve_schrodinger()
        # 3. 计算并显示系数
        # 4. 绘制演化图
        # 5. 验证概率守恒
        # 6. 创建动画
        print("开始求解薛定谔方程...")
        self.solve_schrodinger() # 求解薛定谔方程
        print("计算透射和反射系数...")
        T, R = self.calculate_coefficients() # 计算透射和反射系数
        print(f"透射系数 T = {T:.4f}, 反射系数 R = {R:.4f}")
        print("绘制波函数演化图...")
        self.plot_evolution() # 绘制波函数演化图
        print("验证概率守恒...")
        probabilities = self.verify_probability_conservation() # 验证概率守恒
        print("每个时间步的总概率:", probabilities)
        print("创建波包演化动画...")
        ani = self.create_animation()
        plt.show()
        # 显示动画
        print("演示完成！")


def demonstrate_quantum_tunneling():
    """便捷的演示函数
    
    创建默认参数的求解器并运行演示
    
    返回:
        animation对象
    """
    # TODO: 创建求解器实例并调用demonstrate方法
    solver = QuantumTunnelingSolver()  # 使用默认参数创建求解器实例
    return solver.demonstrate()  # 调用演示方法并返回动画对象


if __name__ == "__main__":
    # 运行演示
    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()
