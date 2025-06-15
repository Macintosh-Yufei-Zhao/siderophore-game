import numpy as np
from numba import cuda, float64
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import time

# 系统参数
alpha = 200.0
a_t = 40.0
alpha_0 = 1.0
c_d = 100.0
e = 50.0
f = 50.0
A = 50.0
omega = np.pi  # 修改为π

# 空间网格参数 (控制体积中心)
N1, N2, N3 = 256, 256, 256  # 修改为256^3网格
x1_edges = np.linspace(0, 80, N1+1)   # x1范围0-80
x2_edges = np.linspace(0, 200, N2+1)  # x2范围0-200
x3_edges = np.linspace(0, 200, N3+1)  # x3范围0-200
dx1 = x1_edges[1] - x1_edges[0]
dx2 = x2_edges[1] - x2_edges[0]
dx3 = x3_edges[1] - x3_edges[0]

# 时间参数
CFL = 0.25
dx_min = min(dx1, dx2, dx3)
dt = CFL * dx_min**2  # 自动计算时间步长
dt=0.001
total_time = 0.2
n_steps = int(total_time / dt)
save_every = 1  # 保存间隔

# 反应矩阵
V = np.array([
    [1, -1, 0, 0, 0],
    [0, 0, 1, -1, 0],
    [0, 0, 0, 1, -1]
], dtype=np.float64)

# 有限体积通量计算函数
@cuda.jit(device=True)
def compute_fluxes(P, x1, x2, x3, t, flux_conv, flux_diff):
    """计算通量（对流+扩散）"""
    # 计算漂移系数
    f = cuda.local.array(3, float64)
    a = cuda.local.array(5, float64)
    compute_a(a, x1, x2, x3, t)
    for i in range(3):
        f[i] = 0.0
        for j in range(5):
            f[i] += V[i,j] * a[j]
    
    # 计算扩散张量
    D = cuda.local.array((3,3), float64)
    compute_D(D, x1, x2, x3, t)
    
    # 对流项通量（迎风格式）
    flux_conv[0] = max(f[0], 0) * P[0] + min(f[0], 0) * P[1]
    flux_conv[1] = max(f[1], 0) * P[0] + min(f[1], 0) * P[1]
    flux_conv[2] = max(f[2], 0) * P[0] + min(f[2], 0) * P[1]
    
    # 扩散项通量（中心差分）
    flux_diff[0] = -D[0,0] * (P[1] - P[0]) / dx1
    flux_diff[1] = -D[1,1] * (P[1] - P[0]) / dx2
    flux_diff[2] = -D[2,2] * (P[1] - P[0]) / dx3

@cuda.jit(device=True)
def compute_a(a_out, x1, x2, x3, t):
    """计算反应速率"""
    discriminant = (a_t - x2 - 1.0)**2 + 4.0 * a_t
    af = 0.5 * (a_t - x2 - 1.0 + math.sqrt(discriminant))
    
    s_val = A * (1.0 + math.sin(omega * t)) / c_d
    numerator = (e*f - 1)*s_val + 1
    denominator = ((f-1)*s_val + 1) * ((e-1)*s_val + 1)
    g_val = numerator / denominator
    
    a_out[0] = alpha * af / a_t
    a_out[1] = x1
    a_out[2] = alpha_0 * g_val * x1
    a_out[3] = x2
    a_out[4] = x3

@cuda.jit(device=True)
def compute_D(D_out, x1, x2, x3, t):
    """计算扩散张量"""
    a = cuda.local.array(5, float64)
    compute_a(a, x1, x2, x3, t)
    for i in range(3):
        for k in range(3):
            D_out[i,k] = 0.0
            for j in range(5):
                D_out[i,k] += V[i,j] * V[k,j] * a[j]

@cuda.jit
def fvm_kernel(P, P_new, t):
    i, j, k = cuda.grid(3)
    
    if 1 <= i < N1-1 and 1 <= j < N2-1 and 1 <= k < N3-1:
        # 读取相邻单元值
        P_x1 = (P[i-1,j,k], P[i,j,k], P[i+1,j,k])
        P_x2 = (P[i,j-1,k], P[i,j,k], P[i,j+1,k])
        P_x3 = (P[i,j,k-1], P[i,j,k], P[i,j,k+1])
        
        # 计算位置坐标
        x1 = x1_edges[i] + 0.5*dx1
        x2 = x2_edges[j] + 0.5*dx2
        x3 = x3_edges[k] + 0.5*dx3
        
        # 初始化通量数组
        flux_conv = cuda.local.array(3, float64)
        flux_diff = cuda.local.array(3, float64)
        F_x1 = cuda.local.array(2, float64)
        F_x2 = cuda.local.array(2, float64)
        F_x3 = cuda.local.array(2, float64)
        
        # x方向通量
        compute_fluxes((P_x1[0], P_x1[1]), x1-0.5*dx1, x2, x3, t, flux_conv, flux_diff)
        F_x1[0] = (flux_conv[0] + flux_diff[0]) * dx2*dx3
        
        compute_fluxes((P_x1[1], P_x1[2]), x1+0.5*dx1, x2, x3, t, flux_conv, flux_diff)
        F_x1[1] = (flux_conv[0] + flux_diff[0]) * dx2*dx3
        
        # y方向通量
        compute_fluxes((P_x2[0], P_x2[1]), x1, x2-0.5*dx2, x3, t, flux_conv, flux_diff)
        F_x2[0] = (flux_conv[1] + flux_diff[1]) * dx1*dx3
        
        compute_fluxes((P_x2[1], P_x2[2]), x1, x2+0.5*dx2, x3, t, flux_conv, flux_diff)
        F_x2[1] = (flux_conv[1] + flux_diff[1]) * dx1*dx3
        
        # z方向通量
        compute_fluxes((P_x3[0], P_x3[1]), x1, x2, x3-0.5*dx3, t, flux_conv, flux_diff)
        F_x3[0] = (flux_conv[2] + flux_diff[2]) * dx1*dx2
        
        compute_fluxes((P_x3[1], P_x3[2]), x1, x2, x3+0.5*dx3, t, flux_conv, flux_diff)
        F_x3[1] = (flux_conv[2] + flux_diff[2]) * dx1*dx2
        
        # 更新方程
        P_new[i,j,k] = P[i,j,k] - dt/(dx1*dx2*dx3) * (
            (F_x1[1] - F_x1[0]) + 
            (F_x2[1] - F_x2[0]) + 
            (F_x3[1] - F_x3[0])
        )

def main():
    # 初始化概率分布
    P_host = np.zeros((N1, N2, N3), dtype=np.float64)
    
    # 高斯初始条件（中心在(20,70,70)）
    x1_center, x2_center, x3_center = 20, 70, 70
    sigma = 5  # 标准差
    
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                x1 = x1_edges[i] + 0.5*dx1
                x2 = x2_edges[j] + 0.5*dx2
                x3 = x3_edges[k] + 0.5*dx3
                exponent = ((x1-x1_center)**2 + 
                          (x2-x2_center)**2 + 
                          (x3-x3_center)**2) / (2*sigma**2)
                P_host[i,j,k] = math.exp(-exponent)
    
    # 归一化
    norm = np.sum(P_host) * dx1*dx2*dx3
    P_host /= norm
    
    # 分配GPU内存
    d_P = cuda.to_device(P_host)
    d_P_new = cuda.to_device(P_host.copy())
    
    # CUDA网格配置
    threads = (8, 8, 4)
    blocks = (
        (N1 + threads[0]-1) // threads[0],
        (N2 + threads[1]-1) // threads[1],
        (N3 + threads[2]-1) // threads[2]
    )
    
    # 存储时间演化数据
    snapshots = []
    time_points = []
    
    # 主循环
    print("开始模拟...")
    start_time = time.time()
    
    for step in range(n_steps):
        current_t = step * dt
        
        # 执行核函数
        fvm_kernel[blocks, threads](d_P, d_P_new, current_t)
        d_P, d_P_new = d_P_new, d_P
        
        # 定期保存和归一化
        if step % save_every == 0:
            P_host = d_P.copy_to_host()
            total_p = np.sum(P_host) * dx1*dx2*dx3
            P_host /= total_p
            d_P = cuda.to_device(P_host)
            
            # 保存投影
            projection = np.sum(P_host, axis=1) * dx2
            snapshots.append(projection)
            time_points.append(current_t)
            print(f"进度 {step}/{n_steps} | 时间 {current_t:.2f}s | 总概率 {total_p:.6f}")
            np.savetxt('dataout\\'+str(step)+'_output.csv', projection, delimiter=',', fmt='%.6f')

    # 生成三维动态视频
    X1, X3 = np.meshgrid(
        x1_edges[:-1] + 0.5*dx1,
        x3_edges[:-1] + 0.5*dx3,
        indexing='ij'
    )
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel("P(x1,x3)")
    
    def update(frame):
        ax.clear()
        surf = ax.plot_surface(
            X1, X3, snapshots[frame].T,
            cmap='viridis',
            rstride=4,
            cstride=4,
            antialiased=True
        )
        ax.set_zlim(-np.max(snapshots)*1.1, np.max(snapshots)*1.1)
        ax.set_title(f"时间演化 t = {time_points[frame]:.2f}s")
        ax.set_xlabel("x1")
        ax.set_ylabel("x3")
        return surf,
    
    ani = FuncAnimation(fig, update, frames=len(snapshots), blit=False)
    ani.save('evolution_3d.mp4', fps=15, dpi=150)
    
    print(f"模拟完成，总耗时 {time.time()-start_time:.2f} 秒")

if __name__ == "__main__":
    main()