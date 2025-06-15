using Printf, ProgressBars, Plots, LinearAlgebra, FFTW

# 参数设置
const α = 200.0
const a_t = 40.0
const α0 = 1.0
const c_d = 100.0
const e = 50.0
const f = 50.0
const A = 50.0
const ω = π

# 空间离散化 (高分辨率)
const Nx1 = 256
const Nx2 = 256
const Nx3 = 256
const x1_min, x1_max = 0.0, 100.0
const x2_min, x2_max = 0.0, 200.0
const x3_min, x3_max = 0.0, 200.0

const Δx1 = (x1_max - x1_min) / (Nx1 - 1)
const Δx2 = (x2_max - x2_min) / (Nx2 - 1)
const Δx3 = (x3_max - x3_min) / (Nx3 - 1)

const x1 = range(x1_min, x1_max, length=Nx1)
const x2 = range(x2_min, x2_max, length=Nx2)
const x3 = range(x3_min, x3_max, length=Nx3)

# 时间参数
const Δt = 0.005
const T_total = 5.0
const Nt = Int(ceil(T_total / Δt))
const tspan = range(0.0, T_total, length=Nt+1)

# 反应矩阵
const v = [1.0 -1.0 0.0 0.0 0.0;
           0.0 0.0 1.0 -1.0 0.0;
           0.0 0.0 0.0 1.0 -1.0]

# 预计算a1 (依赖于x2)
a1 = @. α/(2a_t) * (a_t - x2 - 1 + sqrt((a_t - x2 - 1)^2 + 4a_t))
a1 = reshape(a1, (1, Nx2, 1))

# 初始化概率分布
P = zeros(Float64, Nx1, Nx2, Nx3)
idx_x1 = argmin(abs.(x1 .- 20.0))
idx_x2 = argmin(abs.(x2 .- 70.0))
idx_x3 = argmin(abs.(x3 .- 70.0))
P[idx_x1, idx_x2, idx_x3] = 1.0
P ./= sum(P)  # 归一化

# 存储二维结果 (x1-x3平面)
prob_x1x3 = zeros(Float64, Nx1, Nx3, Nt+1)
prob_x1x3[:, :, 1] = dropdims(sum(P, dims=2), dims=2)

# 预分配内存
f_i = zeros(Float64, Nx1, Nx2, Nx3, 3)
D_ik = zeros(Float64, Nx1, Nx2, Nx3, 3, 3)
flux_x1 = zeros(Float64, Nx1+1, Nx2, Nx3)
flux_x2 = zeros(Float64, Nx1, Nx2+1, Nx3)
flux_x3 = zeros(Float64, Nx1, Nx2, Nx3+1)

# 辅助函数：计算g和a3
function compute_a3(t, x1_vals)
    s = A * (1 + sin(ω * t)) / c_d
    g = ((e*f - 1)*s + 1) / (((f-1)*s + 1)*((e-1)*s + 1))
    return α0 * g * x1_vals
end

# 计算扩散张量D_ik
function compute_diffusion_tensor!(D_ik, a_vals)
    fill!(D_ik, 0.0)
    for j in 1:5
        for i in 1:3, k in 1:3
            @views D_ik[:, :, :, i, k] .+= v[i, j] * v[k, j] .* a_vals[j]
        end
    end
end

# 主模拟循环
@time for n in ProgressBar(1:Nt)
    t = tspan[n+1]
    
    # 计算a3 (依赖于时间和x1)
    a3 = compute_a3(t, reshape(x1, (Nx1, 1, 1)))
    
    # 计算反应速率
    a_vals = [
        a1,                   # a1
        reshape(x1, (Nx1,1,1)), # a2 = x1
        a3,                   # a3
        reshape(x2, (1,Nx2,1)), # a4 = x2
        reshape(x3, (1,1,Nx3))  # a5 = x3
    ]
    
    # 计算漂移项 f_i
    fill!(f_i, 0.0)
    for j in 1:5
        for i in 1:3
            @views f_i[:, :, :, i] .+= v[i, j] .* a_vals[j]
        end
    end
    
    # 计算扩散张量 D_ik
    compute_diffusion_tensor!(D_ik, a_vals)
    
    # 重置通量
    fill!(flux_x1, 0.0)
    fill!(flux_x2, 0.0)
    fill!(flux_x3, 0.0)
    
    # 计算x1方向通量 (漂移+扩散)
    for i in 2:Nx1, j in 1:Nx2, k in 1:Nx3
        # 界面索引
        i_face = i
        
        # 漂流通量 (迎风格式)
        f1_face = f_i[i-1, j, k, 1] + f_i[i, j, k, 1]
        if f1_face > 0
            flux_drift = f1_face * P[i-1, j, k]
        else
            flux_drift = f1_face * P[i, j, k]
        end
        
        # 扩散通量 (中心差分)
        dPdx1 = (P[i, j, k] - P[i-1, j, k]) / Δx1
        dPdx2 = (P[i, j+1, k] - P[i, j-1, k] + P[i-1, j+1, k] - P[i-1, j-1, k]) / (4Δx2)
        dPdx3 = (P[i, j, k+1] - P[i, j, k-1] + P[i-1, j, k+1] - P[i-1, j, k-1]) / (4Δx3)
        
        # 扩散系数在界面处取平均
        D11 = 0.5 * (D_ik[i-1, j, k, 1, 1] + D_ik[i, j, k, 1, 1])
        D12 = 0.5 * (D_ik[i-1, j, k, 1, 2] + D_ik[i, j, k, 1, 2])
        D13 = 0.5 * (D_ik[i-1, j, k, 1, 3] + D_ik[i, j, k, 1, 3])
        
        flux_diff = -0.5 * (D11 * dPdx1 + D12 * dPdx2 + D13 * dPdx3)
        
        flux_x1[i_face, j, k] = flux_drift + flux_diff
    end
    
    # 计算x2方向通量 (类似x1)
    for i in 1:Nx1, j in 2:Nx2, k in 1:Nx3
        j_face = j
        
        f2_face = f_i[i, j-1, k, 2] + f_i[i, j, k, 2]
        if f2_face > 0
            flux_drift = f2_face * P[i, j-1, k]
        else
            flux_drift = f2_face * P[i, j, k]
        end
        
        dPdx1 = (P[i+1, j, k] - P[i-1, j, k] + P[i+1, j-1, k] - P[i-1, j-1, k]) / (4Δx1)
        dPdx2 = (P[i, j, k] - P[i, j-1, k]) / Δx2
        dPdx3 = (P[i, j, k+1] - P[i, j, k-1] + P[i, j-1, k+1] - P[i, j-1, k-1]) / (4Δx3)
        
        D21 = 0.5 * (D_ik[i, j-1, k, 2, 1] + D_ik[i, j, k, 2, 1])
        D22 = 0.5 * (D_ik[i, j-1, k, 2, 2] + D_ik[i, j, k, 2, 2])
        D23 = 0.5 * (D_ik[i, j-1, k, 2, 3] + D_ik[i, j, k, 2, 3])
        
        flux_diff = -0.5 * (D21 * dPdx1 + D22 * dPdx2 + D23 * dPdx3)
        
        flux_x2[i, j_face, k] = flux_drift + flux_diff
    end
    
    # 计算x3方向通量 (类似x1)
    for i in 1:Nx1, j in 1:Nx2, k in 2:Nx3
        k_face = k
        
        f3_face = f_i[i, j, k-1, 3] + f_i[i, j, k, 3]
        if f3_face > 0
            flux_drift = f3_face * P[i, j, k-1]
        else
            flux_drift = f3_face * P[i, j, k]
        end
        
        dPdx1 = (P[i+1, j, k] - P[i-1, j, k] + P[i+1, j, k-1] - P[i-1, j, k-1]) / (4Δx1)
        dPdx2 = (P[i, j+1, k] - P[i, j-1, k] + P[i, j+1, k-1] - P[i, j-1, k-1]) / (4Δx2)
        dPdx3 = (P[i, j, k] - P[i, j, k-1]) / Δx3
        
        D31 = 0.5 * (D_ik[i, j, k-1, 3, 1] + D_ik[i, j, k, 3, 1])
        D32 = 0.5 * (D_ik[i, j, k-1, 3, 2] + D_ik[i, j, k, 3, 2])
        D33 = 0.5 * (D_ik[i, j, k-1, 3, 3] + D_ik[i, j, k, 3, 3])
        
        flux_diff = -0.5 * (D31 * dPdx1 + D32 * dPdx2 + D33 * dPdx3)
        
        flux_x3[i, j, k_face] = flux_drift + flux_diff
    end
    
    # 应用零通量边界条件
    # x1边界
    flux_x1[1, :, :] .= 0.0
    flux_x1[end, :, :] .= 0.0
    
    # x2边界
    flux_x2[:, 1, :] .= 0.0
    flux_x2[:, end, :] .= 0.0
    
    # x3边界
    flux_x3[:, :, 1] .= 0.0
    flux_x3[:, :, end] .= 0.0
    
    # 更新概率分布 (有限体积法)
    for i in 1:Nx1, j in 1:Nx2, k in 1:Nx3
        flux_in = 0.0
        flux_out = 0.0
        
        # x1方向净通量
        flux_in += flux_x1[i, j, k]
        flux_out += flux_x1[i+1, j, k]
        
        # x2方向净通量
        flux_in += flux_x2[i, j, k]
        flux_out += flux_x2[i, j+1, k]
        
        # x3方向净通量
        flux_in += flux_x3[i, j, k]
        flux_out += flux_x3[i, j, k+1]
        
        # 更新概率
        net_flux = flux_in - flux_out
        P[i, j, k] += Δt / (Δx1 * Δx2 * Δx3) * net_flux
    end
    
    # 确保概率为正并归一化
    P = max.(P, 0.0)
    total_prob = sum(P)
    if total_prob > 0
        P ./= total_prob
    end
    
    # 保存二维投影
    prob_x1x3[:, :, n+1] = dropdims(sum(P, dims=2), dims=2)
end

# 创建视频 (10秒视频，30fps)
video_fps = 30
frame_skip = max(1, floor(Int, Nt / (10 * video_fps)))
frames = 1:frame_skip:Nt+1

anim = @animate for i in frames
    t_val = tspan[i]
    prob_slice = log10.(max.(prob_x1x3[:, :, i], 1e-10))  # 对数变换增强可视化
    
    heatmap(x1, x3, prob_slice',
        title="Probability Distribution (t = $(@sprintf("%.2f", t_val))s",
        xlabel="x1", ylabel="x3",
        color=:viridis,
        clims=(-10, 0),  # 对数概率范围
        aspect_ratio=:equal,
        dpi=200)
end

gif(anim, "fokker_planck_simulation.gif", fps=video_fps)