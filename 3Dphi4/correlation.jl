using JLD2
using Statistics
using NPZ
using Plots
using FFTW

# ==================== 基础函数 ====================

function diagonality_3d(nx, ny, nz, L)
    """
    Compute diagonality parameter: sum(p_i^4) / (sum(p_i^2))^2

    In 3D:
    - Most diagonal (1,1,1) -> 3/(3^2) = 1/3 ≈ 0.333
    - Most axial (1,0,0) -> 1/1 = 1

    Lower values = more diagonal (better for avoiding lattice artifacts)
    """
    kx = 2π * nx / L
    ky = 2π * ny / L
    kz = 2π * nz / L

    # Lattice momentum components
    px2 = 4 * sin(kx / 2)^2
    py2 = 4 * sin(ky / 2)^2
    pz2 = 4 * sin(kz / 2)^2

    sum_p2 = px2 + py2 + pz2
    sum_p4 = px2^2 + py2^2 + pz2^2

    if sum_p2 < 1e-10  # k=0 case
        return 0.0
    end
    return sum_p4 / sum_p2^2
end

function lattice_k_sq_3d(nx, ny, nz, L)
    kx = 2π * nx / L
    ky = 2π * ny / L
    kz = 2π * nz / L
    # 格点动量（消除色散关系修正）
    return 4 * sin(kx / 2)^2 + 4 * sin(ky / 2)^2 + 4 * sin(kz / 2)^2
end

# ==================== 动量空间传播子 ====================

function get_momentum_propagator(cfgs::Array{T,4}) where T
    Lx, Ly, Lz, N = size(cfgs)
    V = Lx * Ly * Lz

    G_k_sum = zeros(Lx, Ly, Lz)

    for i in 1:N
        cfg = @view cfgs[:, :, :, i]
        cfgc = cfg .- mean(cfg)
        # 傅里叶变换
        phi_k = fft(cfgc) ./ sqrt(V)  # 归一化
        # |φ̃(k)|²
        G_k = abs2.(phi_k)
        G_k_sum .+= G_k
    end

    G_k_mean = G_k_sum ./ N
    return G_k_mean
end

function get_momentum_propagator_with_err(cfgs::Array{T,4}) where T
    """返回每个k点的传播子均值和标准误差"""
    Lx, Ly, Lz, N = size(cfgs)
    V = Lx * Ly * Lz

    # 存储每个构型的G(k)
    G_k_all = zeros(Lx, Ly, Lz, N)

    for i in 1:N
        cfg = @view cfgs[:, :, :, i]
        cfgc = cfg .- mean(cfg)
        phi_k = fft(cfgc) ./ sqrt(V)
        G_k_all[:, :, :, i] = abs2.(phi_k)
    end

    G_k_mean = mean(G_k_all, dims=4)[:, :, :, 1]
    G_k_std = std(G_k_all, dims=4)[:, :, :, 1]
    G_k_err = G_k_std ./ sqrt(N)  # 标准误差

    return G_k_mean, G_k_err
end

# ==================== Jackknife 函数 ====================

function jack1mk(data) # making JK samples
    Ndata = length(data)
    sum = 0.0
    for i = 1:Ndata
        sum += data[i]
    end
    samples = zeros(Ndata)
    for i = 1:Ndata
        samples[i] = (sum - data[i]) / (Ndata - 1)
    end
    return samples
end

function jack1er(samples) # evaluate mean value and error
    Nsamples = length(samples)
    av = 0.0
    er = 0.0
    for i = 1:Nsamples
        av += samples[i]
        er += samples[i]^2
    end
    av /= Nsamples
    er /= Nsamples
    er = sqrt((er - av^2) * (Nsamples - 1))
    return av, er
end

function get_momentum_propagator_jackknife(cfgs::Array{T,4}) where T
    """
    构造并返回每个 k 点的 jackknife 样本 G_k_jk[:, :, :, i]
    G_k_jk[:, :, :, i] = 去掉第 i 个构型后的传播子平均
    """
    Lx, Ly, Lz, N = size(cfgs)
    V = Lx * Ly * Lz

    # 每个构型的 G_k^{(i)}
    G_k_all = zeros(Float64, Lx, Ly, Lz, N)

    for i in 1:N
        cfg = @view cfgs[:, :, :, i]
        cfgc = cfg .- mean(cfg)
        phi_k = fft(cfgc) ./ sqrt(V)
        G_k_all[:, :, :, i] = abs2.(phi_k)
    end

    # 全样本平均
    G_k_full = mean(G_k_all, dims=4)[:, :, :, 1]

    # jackknife 样本
    G_k_jk = zeros(Float64, Lx, Ly, Lz, N)

    for i in 1:N
        G_k_jk[:, :, :, i] =
            (N .* G_k_full .- G_k_all[:, :, :, i]) ./ (N - 1)
    end

    return G_k_jk
end

# ==================== 径向平均 ====================

function radial_average_3d(G_k::Array{T,3}; max_diagonality=0.34) where {T}
    Lx, Ly, Lz = size(G_k)
    L = Lx

    k_sq_list = Float64[]
    G_list = Float64[]

    n_total = 0
    n_filtered = 0

    for nx in 0:L-1, ny in 0:L-1, nz in 0:L-1
        mx = nx > L ÷ 2 ? nx - L : nx
        my = ny > L ÷ 2 ? ny - L : ny
        mz = nz > L ÷ 2 ? nz - L : nz

        n_total += 1

        # Diagonality filter
        diag = diagonality_3d(mx, my, mz, L)
        if diag > max_diagonality
            n_filtered += 1
            continue
        end

        k_sq = lattice_k_sq_3d(mx, my, mz, L)
        push!(k_sq_list, k_sq)
        push!(G_list, G_k[nx+1, ny+1, nz+1])
    end

    sorted_idx = sortperm(k_sq_list)
    k_sq_sorted = k_sq_list[sorted_idx]
    G_sorted = G_list[sorted_idx]

    unique_k_sq = unique(round.(k_sq_sorted, digits=5))
    k_vals = Float64[]
    G_avg = Float64[]
    counts = Int[]

    for ksq in unique_k_sq
        mask = abs.(k_sq_sorted .- ksq) .< 1e-4
        vals = G_sorted[mask]
        push!(k_vals, sqrt(ksq))  # |k| = sqrt(k²)
        push!(G_avg, mean(vals))
        push!(counts, length(vals))
    end
    return k_vals, G_avg
end

function radial_average_3d_with_err(G_k::Array{T,3}, G_k_err::Array{T,3}; max_diagonality=0.34) where {T}
    """径向平均，同时传播误差"""
    Lx, Ly, Lz = size(G_k)
    L = Lx

    k_sq_list = Float64[]
    G_list = Float64[]
    err_list = Float64[]

    for nx in 0:L-1, ny in 0:L-1, nz in 0:L-1
        mx = nx > L ÷ 2 ? nx - L : nx
        my = ny > L ÷ 2 ? ny - L : ny
        mz = nz > L ÷ 2 ? nz - L : nz

        diag = diagonality_3d(mx, my, mz, L)
        if diag > max_diagonality
            continue
        end

        k_sq = lattice_k_sq_3d(mx, my, mz, L)
        push!(k_sq_list, k_sq)
        push!(G_list, G_k[nx+1, ny+1, nz+1])
        push!(err_list, G_k_err[nx+1, ny+1, nz+1])
    end

    sorted_idx = sortperm(k_sq_list)
    k_sq_sorted = k_sq_list[sorted_idx]
    G_sorted = G_list[sorted_idx]
    err_sorted = err_list[sorted_idx]

    unique_k_sq = unique(round.(k_sq_sorted, digits=5))
    k_vals = Float64[]
    G_avg = Float64[]
    G_err = Float64[]

    for ksq in unique_k_sq
        mask = abs.(k_sq_sorted .- ksq) .< 1e-4
        vals = G_sorted[mask]
        errs = err_sorted[mask]
        n = length(vals)

        push!(k_vals, sqrt(ksq))
        push!(G_avg, mean(vals))
        # 误差传播: 加权平均的误差 = sqrt(sum(err^2)) / n
        push!(G_err, sqrt(sum(errs.^2)) / n)
    end
    return k_vals, G_avg, G_err
end

function radial_average_3d_jackknife(G_k_jk::Array{T,4}; max_diagonality=0.34) where T
    """
    对 jackknife 的 G(k) 样本做径向平均并计算误差
    """
    Lx, Ly, Lz, N = size(G_k_jk)

    # 第一个样本决定 k bin
    k_vals, _ = radial_average_3d(G_k_jk[:, :, :, 1]; max_diagonality=max_diagonality)
    nb = length(k_vals)

    G_rad_jk = zeros(Float64, nb, N)

    for i in 1:N
        _, Gtmp = radial_average_3d(G_k_jk[:, :, :, i]; max_diagonality=max_diagonality)
        G_rad_jk[:, i] .= Gtmp
    end

    # jackknife 平均
    G_mean = mean(G_rad_jk, dims=2)[:, 1]

    # jackknife 误差
    G_err = zeros(Float64, nb)
    for i in 1:N
        G_err .+= (G_rad_jk[:, i] .- G_mean) .^ 2
    end
    G_err .= sqrt.((N - 1) / N .* G_err)

    return k_vals, G_mean, G_err
end

# ==================== 实空间关联函数 ====================

function get_corr_func_3d(cfgs::Array{T,4}) where T
    Lx, Ly, Lz, N = size(cfgs)
    L = Lx  # 假设立方格子

    # 平均磁化强度的平方（断联部分）
    mag_sq = mean(cfgs)^2

    corr_func = Vector{Vector{Float64}}()

    # 对于周期边界条件，最大有意义距离是 L÷2
    for r in 1:(L ÷ 2)
        corrs = zeros(N)

        # 在三个空间方向上计算关联并平均
        for sample_idx in 1:N
            cfg = @view cfgs[:, :, :, sample_idx]

            # x方向位移
            corr_x = mean(cfg .* circshift(cfg, (r, 0, 0)))
            # y方向位移
            corr_y = mean(cfg .* circshift(cfg, (0, r, 0)))
            # z方向位移
            corr_z = mean(cfg .* circshift(cfg, (0, 0, r)))

            # 三个方向平均
            corrs[sample_idx] = (corr_x + corr_y + corr_z) / 3.0
        end

        # 减去断联部分
        corr_mean = mean(corrs .- mag_sq)
        push!(corr_func, [Float64(r), corr_mean])
    end

    return reduce(hcat, corr_func)'  # 转换为矩阵形式
end

# ==================== 自相关与 Binning 分析 ====================

function autocorrelation(data::Vector{T}, max_lag::Int) where T
    """
    计算自相关函数 C(τ) = <(x_t - μ)(x_{t+τ} - μ)> / σ²
    返回 lag=0 到 lag=max_lag 的自相关系数
    """
    n = length(data)
    mean_val = mean(data)
    var_val = var(data, corrected=false)

    if var_val < 1e-15
        return ones(max_lag + 1)  # 常数序列
    end

    acf = zeros(max_lag + 1)
    for lag in 0:max_lag
        if lag >= n
            break
        end
        acf[lag+1] = mean((data[1:n-lag] .- mean_val) .* (data[1+lag:n] .- mean_val)) / var_val
    end
    return acf
end

function integrated_autocorr_time(acf::Vector{Float64}; cutoff=0.05)
    """
    计算积分自相关时间 τ_int = 1/2 + Σ_{t=1}^{t_max} ρ(t)
    cutoff: 当 acf < cutoff 时截断求和
    """
    τ_int = 0.5  # 从 1/2 开始
    for i in 2:length(acf)
        if acf[i] < cutoff
            break
        end
        τ_int += acf[i]
    end
    return τ_int
end

function binning_analysis(data::Vector{T}, max_bin_size::Int=0) where T
    """
    Binning 分析：检测自相关导致的误差低估

    返回:
    - bin_sizes: bin 大小列表
    - errors: 对应的标准误差估计
    - effective_n: 有效样本数估计

    如果误差随 bin size 增大而增大并趋于平台，说明存在自相关
    平台处的误差是真正的统计误差
    """
    n = length(data)
    if max_bin_size == 0
        max_bin_size = n ÷ 10  # 默认最大 bin size
    end

    errors = Float64[]
    bin_sizes = Int[]

    for bin_size in 1:max_bin_size
        n_bins = n ÷ bin_size
        if n_bins < 10  # 至少需要 10 个 bin 来估计误差
            break
        end

        # 计算 binned 数据的均值
        binned = zeros(n_bins)
        for i in 1:n_bins
            binned[i] = mean(data[(i-1)*bin_size+1:i*bin_size])
        end

        # 标准误差 = std(binned means) / sqrt(n_bins)
        push!(errors, std(binned) / sqrt(n_bins))
        push!(bin_sizes, bin_size)
    end

    # 估计有效样本数: N_eff = N * (σ_naive / σ_binned)^2
    naive_err = std(data) / sqrt(n)
    if length(errors) > 0 && errors[end] > 0
        effective_n = n * (naive_err / errors[end])^2
    else
        effective_n = Float64(n)
    end

    return bin_sizes, errors, effective_n
end

function analyze_propagator_autocorrelation(cfgs::Array{T,4};
                                            k_indices=[(2,1,1), (1,2,1), (1,1,2)],
                                            max_lag::Int=50) where T
    """
    分析传播子样本的自相关性

    参数:
    - cfgs: 场构型数组 (Lx, Ly, Lz, N)
    - k_indices: 要分析的 k 点索引列表（1-based）
    - max_lag: 最大自相关滞后

    返回每个 k 点的自相关函数和积分自相关时间
    """
    Lx, Ly, Lz, N = size(cfgs)
    V = Lx * Ly * Lz

    # 计算每个样本的 G(k)
    G_k_all = zeros(Float64, Lx, Ly, Lz, N)
    for i in 1:N
        cfg = @view cfgs[:, :, :, i]
        cfgc = cfg .- mean(cfg)
        phi_k = fft(cfgc) ./ sqrt(V)
        G_k_all[:, :, :, i] = abs2.(phi_k)
    end

    results = Dict()

    for (ix, iy, iz) in k_indices
        # 提取该 k 点的时间序列
        Gk_series = G_k_all[ix, iy, iz, :]

        # 计算自相关
        acf = autocorrelation(Gk_series, min(max_lag, N-1))
        τ_int = integrated_autocorr_time(acf)

        # Binning 分析
        bin_sizes, bin_errors, n_eff = binning_analysis(Gk_series)

        results[(ix, iy, iz)] = (
            acf=acf,
            τ_int=τ_int,
            bin_sizes=bin_sizes,
            bin_errors=bin_errors,
            n_eff=n_eff,
            naive_err=std(Gk_series)/sqrt(N)
        )
    end

    return results
end

function plot_autocorrelation_analysis(results; k_label="k")
    """绘制自相关分析结果"""
    p1 = plot(title="Autocorrelation Function", xlabel="Lag", ylabel="ρ(τ)")
    p2 = plot(title="Binning Analysis", xlabel="Bin Size", ylabel="Standard Error")

    for (k_idx, res) in results
        label = "$k_label=$k_idx"
        plot!(p1, 0:length(res.acf)-1, res.acf, label=label * " (τ_int=$(round(res.τ_int, digits=2)))")
        plot!(p2, res.bin_sizes, res.bin_errors, label=label, marker=:circle, markersize=2)
        hline!(p2, [res.naive_err], linestyle=:dash, label="", alpha=0.5)
    end

    plot(p1, p2, layout=(1,2), size=(1000, 400))
end

function corrected_jackknife_error(G_err::Vector{Float64}, τ_int::Float64)
    """
    使用积分自相关时间修正 jackknife 误差
    真实误差 ≈ jackknife误差 × sqrt(2τ_int)
    """
    return G_err .* sqrt(2 * τ_int)
end

# ==================== 辅助函数 ====================

function khat2_array(L)
    kh2 = zeros(Float64, L, L, L)
    for nx in 0:L-1, ny in 0:L-1, nz in 0:L-1
        mx = nx > L ÷ 2 ? nx - L : nx
        my = ny > L ÷ 2 ? ny - L : ny
        mz = nz > L ÷ 2 ? nz - L : nz
        kx, ky, kz = 2π * mx / L, 2π * my / L, 2π * mz / L
        kh2[nx+1, ny+1, nz+1] = 4sin(kx / 2)^2 + 4sin(ky / 2)^2 + 4sin(kz / 2)^2
    end
    kh2
end

function mags(cfgs)
    N = size(cfgs, 4)
    [mean(@view cfgs[:, :, :, i]) for i in 1:N]
end

# ==================== 主程序示例 ====================

# 加载数据

cfgs = load("data/cfgs_k=0.5_l=2.5_64^3_t=10.jld2")["cfgs"]
# cfgs_dm = npzread("data/phi4_3d_samples.npy")
cfgs32 = load("data/cfgs_k=0.5_l=2.5_32^3_t=10.jld2")["cfgs"]
# 使用 jackknife 计算带误差的传播子
println("Computing jackknife samples...")

# G_k_jk = get_momentum_propagator_jackknife(cat(cfgs32[:, :, :, 1+500:512+500], -cfgs32[:, :, :, 1+500:512+500], dims=4))
G_k_jk = get_momentum_propagator_jackknife(cfgs32)
k_vals, G_mean, G_err = radial_average_3d_jackknife(G_k_jk, max_diagonality=0.34)

# writedlm("data/G_k_32_jk_all.dat", [k_vals G_mean G_err])
# writedlm("data/G_k_32_jk_512*2.dat", [k_vals G_mean G_err])

# 绘图
plot(k_vals[2:end], G_mean[2:end], yerr=G_err[2:end],
     seriestype=:scatter, xaxis=:log, xlabel="p", ylabel="G(p)",
     title="64^3, Broken phase (Jackknife)", label="Data", markerstrokecolor=:auto)


using DelimitedFiles
# writedlm("data/G_k_64_jk.dat", [k_vals G_mean G_err])


k_vals=readdlm("data/G_k_64_jk.dat")[:,1]
G_mean=readdlm("data/G_k_64_jk.dat")[:,2]
G_err=readdlm("data/G_k_64_jk.dat")[:,3]
plot(k_vals[2:end], G_mean[2:end], yerr=G_err[2:end],
     seriestype=:scatter, xaxis=:log, xlabel="p", ylabel="G(p)",
     title="64^3, Broken phase (Jackknife)", label="Data", markerstrokecolor=:auto)

# cfgs_dm64 = cat(npzread("3Dphi4/data/phi4_3d_L32_k0.5_l2.5_ncsnpp_t=10.npy"), npzread("3Dphi4/data/phi4_3d_L32_k0.5_l2.5_ncsnpp_t=10_2.npy"), dims=4)
cfgs_dm64 = npzread("3Dphi4/data/phi4_3d_L32_k0.5_l2.5_ncsnpp_t=10_tau=0.7.npy")
# cfgs_dm64 = npzread("3Dphi4/data/phi4_3d_L64_k0.5_l2.5_t=10.npy")

G_k_jk_dm64 = get_momentum_propagator_jackknife(cfgs_dm64)
k_vals_dm64, G_mean_dm64, G_err_dm64 = radial_average_3d_jackknife(G_k_jk_dm64, max_diagonality=0.34)
plot!(k_vals_dm64[2:end], G_mean_dm64[2:end], yerr=G_err_dm64[2:end],
     seriestype=:scatter, xaxis=:log, xlabel="p", ylabel="G(p)",
     title="64^3, Broken phase (Jackknife)", label="DM", markerstrokecolor=:auto)



writedlm("data/G_k_32_jk_DM_512_2_tau=0.85.dat", [k_vals_dm64 G_mean_dm64 G_err_dm64])
# ==================== 自相关分析示例 ====================

# 对 DM 样本进行自相关分析
println("Analyzing autocorrelation for DM samples...")

# 选择几个代表性的 k 点进行分析（低 k 通常自相关更强）
k_points = [(2,1,1), (3,1,1), (2,2,1), (4,1,1)]  # 1-based 索引

results = analyze_propagator_autocorrelation(cfgs_dm64, k_indices=k_points, max_lag=100)

# 打印分析结果
println("\n===== Autocorrelation Analysis Results =====")
for (k_idx, res) in results
    println("k = $k_idx:")
    println("  Integrated autocorrelation time τ_int = $(round(res.τ_int, digits=2))")
    println("  Naive error = $(round(res.naive_err, digits=6))")
    println("  Effective sample size N_eff ≈ $(round(res.n_eff, digits=1)) (out of $(size(cfgs_dm64, 4)))")
    println("  Error correction factor = $(round(sqrt(2*res.τ_int), digits=2))")
    println()
end

# 绘制自相关分析图
p_acf = plot_autocorrelation_analysis(results)
savefig(p_acf, "autocorrelation_analysis.png")

# 如果需要修正误差棒
# 取最大的 τ_int 作为保守估计
τ_int_max = maximum([res.τ_int for (_, res) in results])
println("Using τ_int = $τ_int_max for error correction")

G_err_corrected = corrected_jackknife_error(G_err_dm64, τ_int_max)

# 用修正后的误差重新绘图
plot(k_vals_dm64[2:end], G_mean_dm64[2:end], yerr=G_err_corrected[2:end],
     seriestype=:scatter, xaxis=:log, xlabel="p", ylabel="G(p)",
     title="DM with Corrected Errors (τ_int=$(round(τ_int_max, digits=1)))",
     label="DM (corrected)", markerstrokecolor=:auto)
