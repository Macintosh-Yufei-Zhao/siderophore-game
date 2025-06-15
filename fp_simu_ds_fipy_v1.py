import numpy as np
import fipy as fp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# 参数设置
alpha = 200.0
a_t = 40.0
alpha_0 = 1.0
c_d = 100.0
e = 50.0
f_param = 50.0
A = 50.0
omega = np.pi

# 网格设置
nx = 21
dx1 = 100.0 / (nx - 1)
dx2 = 200.0 / (nx - 1)
dx3 = 200.0 / (nx - 1)
mesh = fp.Grid3D(dx=dx1, dy=dx2, dz=dx3, nx=nx, ny=nx, nz=nx)

# 概率分布变量P
P = fp.CellVariable(name="Probability Distribution", mesh=mesh, hasOld=True)

# 初始条件
x1_0, x2_0, x3_0 = 20.0, 70.0, 70.0
sigma = 5.0
x1_vals, x2_vals, x3_vals = mesh.cellCenters

distance_sq = (x1_vals - x1_0)**2 + (x2_vals - x2_0)**2 + (x3_vals - x3_0)**2
P.value = fp.numerix.exp(-distance_sq / (2 * sigma**2))
P.value /= P.value.sum() * mesh.cellVolumes[0]

# 视频设置
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlabel("x1")
ax.set_ylabel("x3")
ax.set_title("x1-x3 Probability Distribution (Integrated over x2)")

nz, ny, nx_dim = nx, nx, nx
x1_3d = x1_vals.reshape((nz, ny, nx_dim))
x3_3d = x3_vals.reshape((nz, ny, nx_dim))
x1_coords = x1_3d[0, 0, :]
x3_coords = x3_3d[:, 0, 0]
X1, X3 = np.meshgrid(x1_coords, x3_coords)

im = ax.pcolormesh(X1, X3, np.zeros((nz, nx_dim)), shading='auto', cmap='viridis')
plt.colorbar(im, label='Probability Density')

# 视频参数（2秒模拟生成10秒视频）
writer = FFMpegWriter(fps=10, metadata=dict(title='FP Evolution'))
output_file = "x1x3_integrated.mp4"

simulation_time = 2.0
dt = 0.01
steps = int(simulation_time / dt)

with writer.saving(fig, output_file, dpi=100):
    for step in range(steps):
        t_current = step * dt
        
        # --- 求解方程 ---
        s_val = A * (1 + fp.numerix.sin(omega * t_current)) / c_d
        numerator = (e * f_param - 1) * s_val + 1
        denominator = ((f_param - 1) * s_val + 1) * ((e - 1) * s_val + 1)
        g_val = numerator / denominator

        a_f_val = 0.5 * (a_t - x2_vals - 1 + fp.numerix.sqrt((a_t - x2_vals - 1)**2 + 4 * a_t))

        a1 = alpha * a_f_val / a_t
        a2 = x1_vals
        a3 = alpha_0 * g_val * x1_vals
        a4 = x2_vals
        a5 = x3_vals

        # 修正：使用算术平均插值
        f1 = (a1 - a2).faceValue  # 直接获取面值
        f2 = (a3 - a4).faceValue
        f3 = (a4 - a5).faceValue

        convection_coeff = fp.FaceVariable(mesh=mesh, rank=1)
        convection_coeff[0] = f1
        convection_coeff[1] = f2
        convection_coeff[2] = f3

        D11 = (a1 + a2).faceValue
        D22 = (a3 + a4).faceValue
        D33 = a5.faceValue

        diffusion_coeff = fp.FaceVariable(mesh=mesh, rank=2)
        diffusion_coeff[0, 0] = D11
        diffusion_coeff[1, 1] = D22
        diffusion_coeff[2, 2] = D33

        eq = (fp.TransientTerm() 
              == -fp.ConvectionTerm(convection_coeff) 
              + 0.5 * fp.DiffusionTerm(diffusion_coeff))
        eq.solve(var=P, dt=dt)
        P.updateOld()
        
        # --- 视频处理 ---
        P_3d = P.value.reshape((nz, ny, nx_dim))
        P_x1x3 = P_3d.sum(axis=1)
        
        im.set_array(P_x1x3.ravel())
        im.autoscale()
        
        time_text = ax.text(0.02, 0.95, f"t = {t_current:.1f}s", 
                          transform=ax.transAxes, color='white')
        
        writer.grab_frame()
        time_text.remove()

print(f"视频已保存至：{output_file}")