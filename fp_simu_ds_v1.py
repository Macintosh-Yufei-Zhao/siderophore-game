import numpy as np
from numba import cuda, float64
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import time
from mpl_toolkits.mplot3d import Axes3D
import math

# 系统参数
alpha = 200.0
a_t = 40.0
alpha_0 = 1.0
c_d = 100.0
e = 50.0
f = 50.0
A = 50.0
omega = np.pi

# 空间网格参数
x1_min, x1_max, N1 = 0.0, 50.0, 201   # 减少网格密度以适应显存
x2_min, x2_max, N2 = 0.0, 150.0, 601
x3_min, x3_max, N3 = 0.0, 150.0, 601
dx1 = (x1_max - x1_min) / (N1 - 1)
dx2 = (x2_max - x2_min) / (N2 - 1)
dx3 = (x3_max - x3_min) / (N3 - 1)

# 时间参数
dt = 0.001       # 增大时间步长以加快计算
total_time = 5.0  # 总模拟时间
n_steps = int(total_time / dt)
save_every = 50   # 每50步保存一帧

# 反应矩阵
V = np.array([
    [1, -1, 0, 0, 0],
    [0, 0, 1, -1, 0],
    [0, 0, 0, 1, -1]
], dtype=np.float64)

# GPU设备函数
@cuda.jit(device=True)
def a_f(x2, t):
    discriminant = (a_t - x2 - 1.0)**2 + 4.0 * a_t
    return 0.5 * (a_t - x2 - 1.0 + math.sqrt(discriminant))

@cuda.jit(device=True)
def s(t):
    return A * (1.0 + math.sin(omega * t)) / c_d

@cuda.jit(device=True)
def g(t):
    s_val = s(t)
    numerator = (e*f - 1)*s_val + 1
    denominator = ((f-1)*s_val + 1) * ((e-1)*s_val + 1)
    return numerator / denominator

@cuda.jit(device=True)
def compute_a(a_out, x1, x2, x3, t):  # 修改为传入输出数组
    a_out[0] = alpha * a_f(x2, t) / a_t
    a_out[1] = x1
    a_out[2] = alpha_0 * g(t) * x1
    a_out[3] = x2
    a_out[4] = x3

@cuda.jit(device=True)
def compute_f(f_out, x1, x2, x3, t):  # 修改为传入输出数组
    a = cuda.local.array(5, float64)   # 预先分配内存
    compute_a(a, x1, x2, x3, t)       # 填充数组
    for i in range(3):
        f_out[i] = 0.0
        for j in range(5):
            f_out[i] += V[i,j] * a[j]

@cuda.jit(device=True)
def compute_D(D_out, x1, x2, x3, t):  # 修改为传入输出数组
    a = cuda.local.array(5, float64)
    compute_a(a, x1, x2, x3, t)
    for i in range(3):
        for k in range(3):
            D_out[i,k] = 0.0
            for j in range(5):
                D_out[i,k] += V[i,j] * V[k,j] * a[j]
# CUDA核函数
@cuda.jit
def update_pdf_kernel(P, P_new, t):
    i, j, k = cuda.grid(3)
    
    # 处理虚拟网格点（镜像边界）
    i_p = i if i < N1 else 2*N1 - i - 1
    j_p = j if j < N2 else 2*N2 - j - 1
    k_p = k if k < N3 else 2*N3 - k - 1
    
    if 1 <= i < N1-1 and 1 <= j < N2-1 and 1 <= k < N3-1:
        # 计算空间坐标
        x1 = x1_min + i_p * dx1
        x2 = x2_min + j_p * dx2
        x3 = x3_min + k_p * dx3
        
        # 获取漂移和扩散项
        """ f = compute_f(x1, x2, x3, t)
        D = compute_D(x1, x2, x3, t) """
        
        f = cuda.local.array(3, float64)  # 预分配内存
        compute_f(f, x1, x2, x3, t)      # 传入数组
        
        D = cuda.local.array((3,3), float64)
        compute_D(D, x1, x2, x3, t)
        
        # 计算一阶导数（中心差分）
        dPdx1 = (P[i+1,j,k] - P[i-1,j,k]) / (2*dx1)
        dPdx2 = (P[i,j+1,k] - P[i,j-1,k]) / (2*dx2)
        dPdx3 = (P[i,j,k+1] - P[i,j,k-1]) / (2*dx3)
        
        # 计算二阶导数
        d2Pdx1dx1 = (P[i+1,j,k] - 2*P[i,j,k] + P[i-1,j,k]) / dx1**2
        d2Pdx2dx2 = (P[i,j+1,k] - 2*P[i,j,k] + P[i,j-1,k]) / dx2**2
        d2Pdx3dx3 = (P[i,j,k+1] - 2*P[i,j,k] + P[i,j,k-1]) / dx3**2
        
        # 计算交叉导数项
        d2Pdx1dx2 = (P[i+1,j+1,k] - P[i+1,j-1,k] - P[i-1,j+1,k] + P[i-1,j-1,k]) / (4*dx1*dx2)
        d2Pdx1dx3 = (P[i+1,j,k+1] - P[i+1,j,k-1] - P[i-1,j,k+1] + P[i-1,j,k-1]) / (4*dx1*dx3)
        d2Pdx2dx3 = (P[i,j+1,k+1] - P[i,j+1,k-1] - P[i,j-1,k+1] + P[i,j-1,k-1]) / (4*dx2*dx3)
        
        # 组合各项
        drift = -(f[0]*dPdx1 + f[1]*dPdx2 + f[2]*dPdx3)
        diffusion = 0.5 * (
            D[0,0]*d2Pdx1dx1 + D[1,1]*d2Pdx2dx2 + D[2,2]*d2Pdx3dx3 +
            2*(D[0,1]*d2Pdx1dx2 + D[0,2]*d2Pdx1dx3 + D[1,2]*d2Pdx2dx3)
        )
        
        # 更新概率密度
        P_new[i,j,k] = P[i,j,k] + dt*(drift + diffusion)
        
        """ # 边界条件（零通量）
        if i == 1 or i == N1-2 or j == 1 or j == N2-2 or k == 1 or k == N3-2:
            P_new[i,j,k] = 0.0 """

def main():
    # 初始化概率分布
    P_host = np.zeros((N1, N2, N3), dtype=np.float64)
    
    """ # 高斯初始条件
    x1_center, x2_center, x3_center = 20, 70, 70
    sigma = 5  # 标准差
    
    # 在CPU上生成初始分布
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                x1 = x1_min + i*dx1
                x2 = x2_min + j*dx2
                x3 = x3_min + k*dx3
                exponent = ((x1-x1_center)**2 + 
                          ((x2-x2_center)/50)**2 + 
                          ((x3-x3_center)/50)**2) / (2*sigma**2)
                P_host[i,j,k] = math.exp(-exponent)
    
    # 归一化处理
    norm = np.sum(P_host) * dx1*dx2*dx3
    P_host /= norm """
    
    P_host[20, 70, 70] = 64.0  # 初始条件
    
    # 分配GPU内存
    d_P = cuda.to_device(P_host)
    d_P_new = cuda.to_device(P_host.copy())
    
    # 配置CUDA网格
    threads_per_block = (8, 8, 4)
    blocks_per_grid = (
        (N1 + threads_per_block[0]-1) // threads_per_block[0],
        (N2 + threads_per_block[1]-1) // threads_per_block[1],
        (N3 + threads_per_block[2]-1) // threads_per_block[2]
    )
    
    # 存储时间演化数据
    snapshots = []
    time_points = []
    
    # 主循环
    print("开始模拟...")
    start_time = time()
    for step in range(n_steps):
        current_t = step * dt
        
        # 执行CUDA核函数
        update_pdf_kernel[blocks_per_grid, threads_per_block](d_P, d_P_new, current_t)
        
        # 交换新旧数组指针
        d_P, d_P_new = d_P_new, d_P
        
        """ # 强制归一化
        if step % 10 == 0:  # 每10步归一化一次
            P_host = d_P.copy_to_host()
            total_p = np.sum(P_host) * dx1 * dx2 * dx3
            P_host /= total_p
            d_P = cuda.to_device(P_host) """
        
        # 定期保存数据
        if step % save_every == 0:
            snapshot = d_P.copy_to_host()
            projection = np.sum(snapshot, axis=1) * dx2  # 沿x2积分
            snapshots.append(projection)
            time_points.append(current_t)
            
            
            # 打印进度
            total_p = np.sum(snapshot) * dx1*dx2*dx3
            print(f"时间步 {step:5d}/{n_steps} | 时间 {current_t:.2f}s | 总概率 {total_p:.6f}")

    # 生成视频
    print("生成视频...")
    X1, X3 = np.meshgrid(
        np.linspace(x1_min, x1_max, N1),
        np.linspace(x3_min, x3_max, N3),
        indexing='ij'
    )
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel("概率密度 P(x1,x3)")
    
    def update(frame):
        ax.clear()
        surf = ax.plot_surface(
            X1, X3, snapshots[frame].T,
            cmap='viridis',
            rstride=2,
            cstride=2,
            antialiased=True
        )
        ax.set_zlim(0, np.max(snapshots)*1.1)
        ax.set_title(f"时间演化 t = {time_points[frame]:.2f}s")
        ax.set_xlabel("x1")
        ax.set_ylabel("x3")
        return surf,
    
    ani = FuncAnimation(fig, update, frames=len(snapshots), blit=False)
    ani.save('3d_evolution.mp4', fps=15, dpi=150, 
            extra_args=['-vcodec', 'libx264', '-crf', '18'])
    
    print(f"模拟完成，总耗时 {time()-start_time:.2f} 秒")

if __name__ == "__main__":
    main()