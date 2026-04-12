using LinearAlgebra, CUDA, Random, DifferentialEquations
using StatsBase
using Plots
using Printf
using JLD2
using ProgressLogging, TerminalLoggers

myT=Float32
limitbound(a, n) =
    if a == n + 1
        1
    elseif a == 0
        n
    else
        a
    end


# function update_langevin_2d!(
#     dσ::AbstractArray{T,3},
#     σ::AbstractArray{T,3},
#     fun,
#     κ::T,
#     λ::T,
# ) where {T}
#     N = size(σ, 1)
#     id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     cind = CartesianIndices(σ)
#     for i = id:blockDim().x*gridDim().x:prod(size(σ))
#         x, y, k = Tuple(cind[i])
#         xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
#         yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
#         @inbounds dσ[x, y, k] =
#             2 * κ * (σ[xp1, y, k] + σ[xm1, y, k] + σ[x, yp1, k] + σ[x, ym1, k] - 4 * σ[x, y, k]) - fun(σ[x, y, k], κ, λ)
#     end
#     return nothing
# end

function update_langevin_2d!(
    dσ::AbstractArray{T,3},
    σ::AbstractArray{T,3},
    fun,
    κ::T,
    λ::T,
) where {T}
    N = size(σ, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(σ)
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        x, y, k = Tuple(cind[i])
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        @inbounds dσ[x, y, k] =
            2 * κ * (σ[xp1, y, k] + σ[xm1, y, k] + σ[x, yp1, k] + σ[x, ym1, k]) - fun(σ[x, y, k], κ, λ)
    end
    return nothing
end



function langevin_2d_loop_GPU(dσ, σ, fun, κ, λ)
    threads = 512
    blocks = 2^8
    @cuda blocks = blocks threads = threads maxregs = 4 update_langevin_2d!(dσ, σ, fun, κ, λ)
end


"""
     modelA_2d_ODE_prob(; u0, device=nothing, ...)

单 GPU Langevin 演化。`device` 指定 GPU 编号（0-based），
为 nothing 时使用当前设备。
"""
function modelA_2d_ODE_prob(;
    u0=error("u0 not provided"),
    tspan=myT.((0.0, 15.0)),
    T=1.0f0,
    κ=0.5,
    λ=0.022,
    solver=DRI1NM(),
    save_start=false,
    save_everystep=false,
    save_end=false,
    abstol=1e-2,
    reltol=1e-2,
    saveat=myT[],
    callback=nothing,
    device=nothing,
    progress=true,
    args...,
)
    if device !== nothing
        CUDA.device!(device)
    end

    function Ufun(x, κ, λ)
        2 * (1 - 2 * λ) * x + 4 * λ * x^3
    end
    function g(du, u, p, t)
        du .= sqrt(2*T)
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_2d_loop_GPU(dσ, σ, Ufun, p[1], p[2])
    end
    sdeprob = SDEProblem(ODEfun_tex, g, u0_GPU, tspan, [κ, λ])
    GC.gc(true)
    CUDA.reclaim()
    solve(
        sdeprob,
        solver,
        save_start=save_start,
        save_everystep=save_everystep,
        save_end=save_end,
        abstol=abstol,
        reltol=reltol,
        saveat=saveat,
        # callback=callback,
        dtmin=0.0001,
        maxiters=10000000,
        progress=progress,
        progress_steps=500,
        progress_name="Langevin SDE (GPU $(device === nothing ? CUDA.device() : device))",
    )
end


"""
     modelA_2d_multigpu(; u0, devices=[0,1], kwargs...)

将副本（u0 的第 3 维）均匀分配到多块 GPU，各自独立跑 Langevin 演化，
最后在 CPU 上拼接末态构型返回。

- `u0`：CPU 数组 (L×L×N)
- `devices`：GPU 编号列表（0-based），如 [2, 3] 表示用第 2、3 号 GPU
- 其余参数透传给 `modelA_2d_ODE_prob`

返回 CPU 上的拼接构型 (L×L×N)。
"""
function modelA_2d_multigpu(;
    u0,
    devices=[0, 1],
    tspan=myT.((0.0, 15.0)),
    T=1.0f0,
    κ=0.5f0,
    λ=0.022f0,
    solver=DRI1NM(),
    save_end=true,
    abstol=1e-2,
    reltol=1e-2,
    kwargs...,
)
    n_dev = length(devices)
    L1, L2, N = size(u0)
    @assert N >= n_dev "n_cfgs ($N) must be >= number of devices ($n_dev)"

    # 均匀分配副本到各 GPU
    chunk_sizes = [div(N, n_dev) + (i <= rem(N, n_dev) ? 1 : 0) for i in 1:n_dev]
    offsets = cumsum([0; chunk_sizes[1:end-1]])

    @info "Multi-GPU Langevin: $N replicas on $(n_dev) GPUs $(devices), chunks = $chunk_sizes"

    results = Vector{Any}(undef, n_dev)

    # 每块 GPU 一个 @async task（CUDA.device! 是 task-local 的）
    @sync for (idx, dev) in enumerate(devices)
        @async begin
            CUDA.device!(dev)
            u0_chunk = u0[:, :, offsets[idx]+1 : offsets[idx]+chunk_sizes[idx]]
            @info "  GPU $dev: $(chunk_sizes[idx]) replicas, starting..."
            sol = modelA_2d_ODE_prob(;
                u0=u0_chunk,
                tspan=tspan,
                T=T, κ=κ, λ=λ,
                solver=solver,
                save_end=save_end,
                save_start=false,
                save_everystep=false,
                abstol=abstol,
                reltol=reltol,
                device=dev,
                progress=true,
                kwargs...,
            )
            results[idx] = Array(sol.u[end])
            @info "  GPU $dev: done."
            GC.gc(true)
            CUDA.reclaim()
        end
    end

    cfgs = cat(results..., dims=3)
    @info "Multi-GPU Langevin complete: $(size(cfgs, 3)) configurations collected."
    return cfgs
end

cfgs = modelA_2d_multigpu(;
    u0=randn(myT, 128, 128, 5120),
    devices=[0,1,2, 3],
    tspan=myT.((0.0, 100000)),
    T=1.0f0,
    κ=0.2707f0,
    λ=0.022f0,
    solver=DRI1NM(),
    save_end=true,
    abstol=1e-2,
    reltol=1e-2,
    progress=true,
)

# cb = SavingCallback(
#     (u, t, integrator) -> reshape(
#         mapslices(x -> cumulant.(Ref(abs.(x)), [1, 2, 3, 4]), Array(u); dims=[1, 2]),
#         (4, LangevinDynamics.M),
#     ),
#     saved_values;
#     # saveat=0.0:2.0:1500.0,
# )
saved_values = SavedValues(Float32, Any)
cb = SavingCallback(
    (u, t, integrator) -> begin
        u_cpu = Array(u)   # 拷到 CPU，强制 CUDA 同步，确保拿到与 sol.u 一致的数据
        ϕ = mean(u_cpu, dims=[1, 2])[1, 1, :]
        return mean(abs.(ϕ))
    end,
    saved_values;
    saveat=0.0:2.0:100.0,
    save_everystep=false,
    save_start=true
)




# ==============================================================================
# 热化诊断 (Thermalization Diagnostics)
# ==============================================================================

"""
     thermalization_check(; L=128, n_cfgs=512, κ, λ=0.022f0, T=1.0f0,
                           tspan=(0.0f0, 300.0f0), dt_save=2.0f0,
                           save_path=nothing)

热启动 vs 冷启动诊断：从两种极端初始条件出发演化，
监控多个观测量是否收敛到一致，确定安全的热化时间。

SavingCallback 内部先 Array(u) 拷贝到 CPU 再计算（强制 CUDA 同步，
确保与 sol.u 一致），sol 不存中间快照以节省 GPU 内存。
返回 (t_arr, sv_hot, sv_cold, sol_hot, sol_cold)。
"""
function thermalization_check(;
    L=128,
    n_cfgs=512,
    κ,
    λ=0.022f0,
    T=1.0f0,
    tspan=(0.0f0, 300.0f0),
    dt_save=2.0f0,
    save_path=nothing,
    abstol=2e-2,
    reltol=2e-2,
)
    t_save_range = tspan[1]:dt_save:tspan[2]
    n_t = length(t_save_range)

    function make_obs_callback()
        sv = SavedValues(Float32, NTuple{4,Float32})
        cb = SavingCallback(
            (u, t, integrator) -> begin
                u_cpu = Array(u)  # 拷到 CPU，强制 CUDA 同步，保证与 sol.u 一致
                ϕ = mean(u_cpu, dims=[1, 2])[1, 1, :]  # 每个副本的 φ̄
                mag = mean(ϕ)       # ⟨φ̄⟩
                m2 = mean(ϕ .^ 2)       # ⟨φ̄²⟩
                m4 = mean(ϕ .^ 4)       # ⟨φ̄⁴⟩
                phi2 = mean(u_cpu .^ 2)  # ⟨φ²⟩ (全格点)
                return (mag, m2, m4, phi2)
            end,
            sv;
            saveat=0.0:100.0:80000.0,
            save_everystep=false,
            save_start=true,
        )
        return sv, cb
    end

    # ---- 热启动 (disordered, φ ≈ 0) ----
    @info "Running HOT start (random φ ≈ 0) ..."
    sv_hot, cb_hot = make_obs_callback()
    u0_hot = randn(myT, L, L, n_cfgs)
    sol_hot = modelA_2d_ODE_prob(;
        u0=u0_hot, tspan=tspan, κ=myT(κ), λ=myT(λ), T=myT(T),
        save_end=false, callback=cb_hot, abstol=abstol, reltol=reltol,
    )
    GC.gc(true)
    CUDA.reclaim()

    # ---- 冷启动 (ordered, φ ≈ +φ₀) ----
    @info "Running COLD start (ordered φ ≈ +3) ..."
    sv_cold, cb_cold = make_obs_callback()
    u0_cold = ones(myT, L, L, n_cfgs) .* 3.0f0
    sol_cold = modelA_2d_ODE_prob(;
        u0=u0_cold, tspan=tspan, κ=myT(κ), λ=myT(λ), T=myT(T),
        save_end=false, callback=cb_cold, abstol=abstol, reltol=reltol,
    )
    GC.gc(true)
    CUDA.reclaim()

    # ---- 提取数据（从 callback 的 saveval，已在 CPU 上计算，与 sol.u 一致）----
    t_arr = sv_hot.t
    mag_hot = [v[1] for v in sv_hot.saveval]
    m2_hot = [v[2] for v in sv_hot.saveval]
    phi2_hot = [v[4] for v in sv_hot.saveval]

    mag_cold = [v[1] for v in sv_cold.saveval]
    m2_cold = [v[2] for v in sv_cold.saveval]
    phi2_cold = [v[4] for v in sv_cold.saveval]

    # Binder 随时间
    binder_hot = [1 - v[3] / (3 * v[2]^2) for v in sv_hot.saveval]
    binder_cold = [1 - v[3] / (3 * v[2]^2) for v in sv_cold.saveval]

    # ---- 绘图 ----
    if save_path !== nothing
        try
            p1 = plot(t_arr, mag_hot, label="hot start", lw=1.5, color=:red,
                xlabel="t", ylabel="⟨|φ̄|⟩", title="Magnetization")
            plot!(p1, t_arr, mag_cold, label="cold start", lw=1.5, color=:blue)

            p2 = plot(t_arr, m2_hot, label="hot", lw=1.5, color=:red,
                xlabel="t", ylabel="⟨φ̄²⟩", title="⟨φ̄²⟩")
            plot!(p2, t_arr, m2_cold, label="cold", lw=1.5, color=:blue)

            p3 = plot(t_arr, binder_hot, label="hot", lw=1.5, color=:red,
                xlabel="t", ylabel="U₄", title="Binder Cumulant")
            plot!(p3, t_arr, binder_cold, label="cold", lw=1.5, color=:blue)

            p4 = plot(t_arr, phi2_hot, label="hot", lw=1.5, color=:red,
                xlabel="t", ylabel="⟨φ²⟩", title="⟨φ²⟩ (site avg)")
            plot!(p4, t_arr, phi2_cold, label="cold", lw=1.5, color=:blue)

            # 相对差异
            rel_mag = abs.(mag_hot .- mag_cold) ./ (0.5 .* abs.(mag_hot .+ mag_cold) .+ 1e-10)
            rel_m2 = abs.(m2_hot .- m2_cold) ./ (0.5 .* abs.(m2_hot .+ m2_cold) .+ 1e-10)
            p5 = plot(t_arr, rel_mag, label="|φ̄|", lw=1.5, color=:red,
                xlabel="t", ylabel="relative diff", title="Hot-Cold Convergence",
                yscale=:log10)
            plot!(p5, t_arr, rel_m2, label="φ̄²", lw=1.5, color=:blue)
            hline!(p5, [0.05], ls=:dash, color=:gray, label="5% threshold")

            fig = plot(p1, p2, p3, p4, p5,
                layout=@layout([a b; c d; e]),
                size=(1000, 900), margin=5Plots.mm)
            mkpath(dirname(save_path))
            savefig(fig, save_path)
            @info "Thermalization plot saved to $save_path"
        catch e
            @warn "Plot failed" exception = e
        end
    end

    # ---- 自动估计热化时间 ----
    # 找到 hot 和 cold 的 ⟨|φ̄|⟩ 相对差异首次 < 5% 的时间
    rel_diff = abs.(mag_hot .- mag_cold) ./ (0.5 .* abs.(mag_hot .+ mag_cold) .+ 1e-10)
    t_therm_idx = findfirst(x -> x < 0.05, rel_diff)
    if t_therm_idx !== nothing
        @info "Estimated thermalization time: t ≈ $(t_arr[t_therm_idx]) (hot-cold |φ̄| diff < 5%)"
    else
        @warn "Hot and cold starts have NOT converged within tspan=$(tspan)! Increase tspan."
    end

    return t_arr, sv_hot, sv_cold, sol_hot, sol_cold
end

# 使用示例（取消注释运行）：
t_arr, sv_hot, sv_cold, sol_hot, sol_cold = thermalization_check(;
    L=128, n_cfgs=128, κ=0.27f0, λ=0.022f0,
    tspan=(0.0f0, 80000.0f0), dt_save=2.0f0,
    save_path="trainingdata/thermalization_L128_k0.27.png"
)
t_arr
stack(sv_hot.saveval)[1,:]
stack(sv_hot.saveval)[1, :]
plot(t_arr, stack(sv_hot.saveval)[1,:])
plot(t_arr, stack(sv_cold.saveval)[1,:])

heatmap(Array(sol_hot.u[end-7])[:,:,1],aspect_ratio=1.0,color=:RdBu)

stack(sv_hot.saveval)[1, end-5]

mean(sol_hot.u[end-7])

# ==============================================================================













# ==============================================================================
# 2D φ⁴ 临界温度 / 临界 κ 寻找
# ==============================================================================
# 从构型 cfgs (L×L×N) 计算观测量：序参量 φ̄ = (1/V)∑_x φ(x)，再对 N 个构型求平均。
# 用于扫描 κ（或 T）定位相变点：Binder 累积量交点、磁化率峰值、序参量下降等。

"""
     observables_2d_phi4(cfgs; skip=0)

从 2D φ⁴ 构型 cfgs (Lx×Ly×N) 计算观测量（与 HMC 一致）：
- φ̄_i = (1/V)∑_{x,y} φ(x,y)_i，再对 N 个构型求统计。
返回：(mag_mean, mag_err, suscept, suscept_err, binder, binder_err)
可选 skip：丢弃前 skip 个构型（热化）。
"""
function observables_2d_phi4(cfgs::AbstractArray{T,3}; skip::Int=0) where T
    Lx, Ly, N = size(cfgs)
    V = Lx * Ly
    cfgs = cfgs[:, :, (skip+1):end]
    N = size(cfgs, 3)
    N <= 1 && error("observables_2d_phi4: need N>1 configs after skip")

    # 每个构型的空间平均 φ̄
    phi_bar = [mean(@view cfgs[:, :, i]) for i in 1:N]
    abs_mag = abs.(phi_bar)

    # 序参量：⟨|φ̄|⟩（用绝对值）
    mag_mean = mean(abs_mag)
    mag_err  = std(abs_mag) / sqrt(N - 1)

    m2 = mean(phi_bar .^ 2)                       # ⟨φ̄²⟩
    m4 = mean(phi_bar .^ 4)                       # ⟨φ̄⁴⟩
    suscept = V * (m2 - mag_mean^2)               # χ = V(⟨φ̄²⟩ - ⟨|φ̄|⟩²)
    binder  = 1 - m4 / (3 * m2^2)                # U₄ = 1 - ⟨φ̄⁴⟩/(3⟨φ̄²⟩²)

    # Jackknife 误差：χ 用 ⟨|φ̄|⟩² 作减法项
    sum_abs  = sum(abs_mag)
    sum_phi2 = sum(phi_bar .^ 2)
    sum_phi4 = sum(phi_bar .^ 4)
    jk_abs_mean = [(sum_abs - abs_mag[i]) / (N - 1) for i in 1:N]
    jk_m2       = [(sum_phi2 - phi_bar[i]^2) / (N - 1) for i in 1:N]
    jk_m4       = [(sum_phi4 - phi_bar[i]^4) / (N - 1) for i in 1:N]
    jk_suscept  = V .* (jk_m2 .- jk_abs_mean.^2)
    jk_binder   = 1 .- jk_m4 ./ (3 .* jk_m2.^2)
    var_suscept = (N - 1) * (mean(jk_suscept.^2) - mean(jk_suscept)^2)
    var_binder  = (N - 1) * (mean(jk_binder.^2) - mean(jk_binder)^2)
    suscept_err = sqrt(max(zero(var_suscept), var_suscept))
    binder_err  = sqrt(max(zero(var_binder), var_binder))

    return mag_mean, mag_err, suscept, suscept_err, binder, binder_err
end

"""
     run_langevin_collect_cfgs(; L=64, n_cfgs=512, κ, λ=0.022f0, T=1.0f0, tspan=(0.0f0, 50.0f0), u0_mag=2.5f0, kwargs...)

运行一次 Langevin 演化，返回末态构型 (L×L×n_cfgs) 及观测量。
u0_mag：初始场均值（对称相用 0，破缺相用正数如 2.5）。
"""
function run_langevin_collect_cfgs(;
    L=64,
    n_cfgs=512,
    κ,
    λ=0.022f0,
    T=1.0f0,
    tspan=(0.0f0, 50.0f0),
    u0_mag=0.0f0,
    abstol=1e-2,
    reltol=1e-2,
    kwargs...,
)
    u0ini = randn(myT, L, L, n_cfgs) .+ myT(u0_mag)
    sol = modelA_2d_ODE_prob(;
        u0=u0ini,
        tspan=tspan,
        κ=myT(κ),
        λ=myT(λ),
        T=myT(T),
        save_end=true,
        abstol=abstol,
        reltol=reltol,
        kwargs...,
    )
    cfgs = Array(sol.u[end])
    obs = observables_2d_phi4(cfgs)
    return cfgs, obs
end

"""
     scan_kappa_for_critical(κ_list; L=64, n_cfgs=512, λ=0.022f0, T=1.0f0,
                              tspan=(0.0f0, 50.0f0), u0_mag=2.5f0,
                              save_dir="scan_kappa", ...)

在给定 κ 列表上扫描，每个 κ 跑一次 Langevin，计算 ⟨|φ̄|⟩、χ、Binder U₄。
结果保存到 `save_dir/` 下：
  - 观测量 CSV + 三合一图
  - 每个 κ 的末态构型 `.jld2`（含 cfgs, κ, λ, L 等元数据）
返回：(κ_list, mag, mag_err, suscept, suscept_err, binder, binder_err)
"""
function scan_kappa_for_critical(κ_list;
    L=64,
    n_cfgs=512,
    λ=0.022f0,
    T=1.0f0,
    tspan=(0.0f0, 50.0f0),
    u0_mag=0.0f0,
    skip=0,
    save_dir="scan_kappa",
    abstol=1e-2,
    reltol=1e-2,
    kwargs...,
)
    mkpath(save_dir)
    nk = length(κ_list)
    mag       = fill(NaN, nk)
    mag_err   = fill(NaN, nk)
    suscept   = fill(NaN, nk)
    suscept_err = fill(NaN, nk)
    binder    = fill(NaN, nk)
    binder_err = fill(NaN, nk)

    for (ik, κ) in enumerate(κ_list)
        @info "scan κ = $κ ($ik/$nk), L=$L, n_cfgs=$n_cfgs"
        try
            cfgs, obs = run_langevin_collect_cfgs(;
                L=L, n_cfgs=n_cfgs, κ=κ, λ=λ, T=T, tspan=tspan,
                u0_mag=u0_mag, abstol=abstol, reltol=reltol, kwargs...,
            )
            mag[ik], mag_err[ik], suscept[ik], suscept_err[ik], binder[ik], binder_err[ik] = obs

            # 保存末态构型
            κ_str = replace(@sprintf("%.5f", κ), "." => "p")
            cfgs_file = joinpath(save_dir, "cfgs_L$(L)_k$(κ_str)_l$(λ)_n$(n_cfgs).jld2")
            jldsave(cfgs_file; cfgs, κ, λ, L, T, tspan, n_cfgs)
            @info "  Saved configs → $cfgs_file"
        catch e
            @warn "κ=$κ failed" exception=e
        end
    end

    # 保存观测量 CSV
    csv_file = joinpath(save_dir, "scan_L$(L)_n$(n_cfgs).csv")
    open(csv_file, "w") do io
        println(io, "kappa,mag,mag_err,suscept,suscept_err,binder,binder_err")
        for i in 1:nk
            println(io, κ_list[i], ",", mag[i], ",", mag_err[i], ",",
                    suscept[i], ",", suscept_err[i], ",", binder[i], ",", binder_err[i])
        end
    end
    @info "Saved observables → $csv_file"

    # 绘图
    try
        p1 = scatter(κ_list, mag, yerr=mag_err, xlabel="κ", ylabel="⟨|φ̄|⟩",
                     label="L=$L", title="Magnetization", markerstrokecolor=:auto)
        p2 = scatter(κ_list, suscept, yerr=suscept_err, xlabel="κ", ylabel="χ",
                     label="L=$L", title="Susceptibility", markerstrokecolor=:auto)
        p3 = scatter(κ_list, binder, yerr=binder_err, xlabel="κ", ylabel="U₄",
                     label="L=$L", title="Binder cumulant", markerstrokecolor=:auto)
        fig = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
        fig_file = joinpath(save_dir, "scan_L$(L)_n$(n_cfgs).png")
        savefig(fig, fig_file)
        @info "Saved plot → $fig_file"
    catch
        @warn "Plots not available or savefig failed, skip figures"
    end

    return κ_list, mag, mag_err, suscept, suscept_err, binder, binder_err
end

"""
     estimate_critical_kappa(κ_list, suscept, suscept_err)

从磁化率曲线估计临界 κ：取 χ 最大的 κ 为 κ_c（简单峰值法）。
返回 (κ_c_est, idx_max)。
"""
function estimate_critical_kappa(κ_list, suscept, suscept_err)
    idx = argmax(suscept)
    return κ_list[idx], idx
end

# 使用示例（取消注释并调整 κ 列表后运行）：
κ_range = range(0.2705f0, 0.271f0; length=3)
κ_list = Float32.(collect(κ_range))
CUDA.device!(3)
κ_vals, mag, mag_e, sus, sus_e, bind, bind_e = scan_kappa_for_critical(κ_list; L=128, n_cfgs=512, tspan=(0.0f0, 20000.0f0), u0_mag=0.0f0, save_path="trainingdata/critical_scan_L128_Langevin")
κ_c, i_c = estimate_critical_kappa(κ_vals, sus, sus_e)
# println("Estimated critical κ_c ≈ $κ_c (from susceptibility peak)")

u0ini = randn(myT, 128, 128,10240)
u0ini=Array(sol.u[end])
sol = modelA_2d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10000.0f0), κ=0.2707f0, λ=0.022f0, T=1.0f0, save_end=true)
using Plots


plot(saved_values.t, saved_values.saveval)
cfgs=Array(sol.u[end])


n_samples = size(cfgs, 3)
n_show = min(12, size(cfgs, 3))
n_rows, n_cols = 3, 4
indices = round.(Int, range(1, n_samples, length=n_show))

plots = [heatmap(cfgs[:, :, indices[i]],
    aspect_ratio=1.0,
    color=:RdBu,
    clims=(-4.7, 4.7),
    title="#$(indices[i])",
    colorbar=false,
    axis=false,
    ticks=false
) for i in 1:n_show]

fig2 = plot(plots...,
    layout=(n_rows, n_cols),
    size=(900, 700),
    plot_title="Sampled Configurations ($n_show snapshots)")


n_samples=size(cfgs, 3)

n_show = min(12, size(cfgs, 3))
n_cols = cld(n_show, 3)
indices = round.(Int, range(1, n_samples, length=n_show))
rows = []
for row in 1:3
    idx_range = ((row-1)*n_cols+1):min(row * n_cols, n_show)
    push!(rows, hcat([cfgs[:, :, i] for i in indices[idx_range]]...))
end
cfg_cat = vcat(rows...)
fig2 = heatmap(cfg_cat,
    aspect_ratio=1.0,
    # clims  = (-4.7, 4.7),
    color=:RdBu,
    title="Sampled Configurations ($n_show snapshots)",
    size=(900, 700))



using DelimitedFiles
kdata=readdlm("trainingdata/critical_scan_L128_Langevin_scan.csv",',')[2:end,1]
Binder_data=readdlm("trainingdata/critical_scan_L128_Langevin_scan.csv",',')[2:end,6]
Binder_err_data=readdlm("trainingdata/critical_scan_L128_Langevin_scan.csv",',')[2:end,7]

plot(kdata, Binder_data, yerr=Binder_err_data, seriestype=:scatter, xlabel="κ", ylabel="Binder", title="Binder")


kdata=readdlm("trainingdata/critical_scan_L64_Langevin_scan.csv",',')[2:end,1]
Binder_data=readdlm("trainingdata/critical_scan_L64_Langevin_scan.csv",',')[2:end,6]
Binder_err_data=readdlm("trainingdata/critical_scan_L64_Langevin_scan.csv",',')[2:end,7]
plot!(kdata, Binder_data, yerr=Binder_err_data, seriestype=:scatter, xlabel="κ", ylabel="Binder", title="Binder")



reshape(sol1,32,32*10)
heatmap(reshape(sol1, 32, 32 * 11), aspect_ratio=1.0, clims=(-5.7, 5.7))
plot(Array(mean(Array(stack(sol.u)), dims=[1, 2, 3])[1, 1, 1, :]))

# plot(mean(abs.(Array(stack(sol.u))), dims=[1, 2, 3])[1, 1, 1, 1:end])


plot!(mean(abs.(mean(Array(stack(sol.u)), dims=[1, 2])), dims=[3])[1, 1, 1, 1:end])
cfgs=reshape(stack(sol.u)[:,:,:,end],32,32,4096*16)|>Array


cfgs = sol.u[end] |> Array

using JLD2

jldsave("DMasSQ-main/data/cfgs_k=0.21_l=0.022.jld2"; cfgs)

jldsave("data/cfgs_k=0.5_l=0.022_16^2_t=10.jld2"; cfgs)

jldsave("data/cfgs_k=0.21_l=0.022_small.jld2"; cfgs)
jldsave("trainingdata/cfgs_k=0.2707_l=0.022_128^2_t=50000.jld2"; cfgs)


let usol = Array(stack(sol.u))
    plot(mean(var(usol,dims=[1,2]),dims=[1,2,3])[1,1,1,:])
end
var

abs.(stack(sol.u))
