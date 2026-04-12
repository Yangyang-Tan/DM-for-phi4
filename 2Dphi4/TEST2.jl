using JLD2
using Statistics
using NPZ
using Plots
using FFTW

function get_momentum_propagator_jackknife(cfgs::Array{T,3}) where T

    Lx, Ly, N = size(cfgs)
    V = Lx * Ly

    # 每个构型的 G_k^{(i)}
    G_k_all = zeros(Float64, Lx, Ly, N)

    for i in 1:N
        cfg = @view cfgs[:, :, i]
        cfgc = cfg .- mean(cfg)
        phi_k = fft(cfgc) ./ sqrt(V)
        G_k_all[:, :, i] = abs2.(phi_k)
    end

    # 全样本平均
    G_k_full = mean(G_k_all, dims=3)[:, :, 1]

    # jackknife 样本
    G_k_jk = zeros(Float64, Lx, Ly, N)

    for i in 1:N
        G_k_jk[:, :, i] =
            (N .* G_k_full .- G_k_all[:, :, i]) ./ (N - 1)
    end

    return G_k_jk
end


function radial_average_2d_jackknife(G_k_jk::Array{T,3};
    max_diagonality=0.51) where T

    Lx, Ly, N = size(G_k_jk)

    # 第一个样本决定 k bin
    k_vals, _ = radial_average_2d(G_k_jk[:, :, 1];
        max_diagonality=max_diagonality)
    nb = length(k_vals)

    G_rad_jk = zeros(Float64, nb, N)

    for i in 1:N
        _, Gtmp = radial_average_2d(G_k_jk[:, :, i];
            max_diagonality=max_diagonality)
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


function lattice_k_sq(nx, ny, L)
    kx = 2π * nx / L
    ky = 2π * ny / L
    # 格点动量（消除色散关系修正）
    return 4 * sin(kx / 2)^2 + 4 * sin(ky / 2)^2
end

function radial_average_2d(G_k::Array{T,2}; max_diagonality=0.51) where {T}
    Lx, Ly = size(G_k)
    L = Lx


    k_sq_list = Float64[]
    G_list = Float64[]

    n_total = 0
    n_filtered = 0

    for nx in 0:L-1, ny in 0:L-1
        mx = nx > L ÷ 2 ? nx - L : nx
        my = ny > L ÷ 2 ? ny - L : ny

        n_total += 1

        # Diagonality filter
        diag = diagonality_2d(mx, my, L)
        if diag > max_diagonality
            n_filtered += 1
            continue
        end

        k_sq = lattice_k_sq(mx, my, L)
        push!(k_sq_list, k_sq)
        push!(G_list, G_k[nx+1, ny+1])
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

function diagonality_2d(nx, ny, L)
    kx = 2π * nx / L
    ky = 2π * ny / L

    # Lattice momentum components
    px2 = 4 * sin(kx / 2)^2
    py2 = 4 * sin(ky / 2)^2

    sum_p2 = px2 + py2
    sum_p4 = px2^2 + py2^2

    if sum_p2 < 1e-10  # k=0 case
        return 0.0
    end
    return sum_p4 / sum_p2^2
end

cfgs = load("2Dphi4/trainingdata/cfgs_hmc_k=0.2707_l=0.022_128^2_tree12.jld2")["cfgs"]

cfgs = load("2Dphi4/trainingdata/cfgs_k=0.2707_l=0.022_128^2_t=10000.jld2")["cfgs"]

G_k_jk = get_momentum_propagator_jackknife(cfgs[:,:,200:end])
k_vals, G_mean, G_err = radial_average_2d_jackknife(G_k_jk, max_diagonality=0.51)
plot!(k_vals[2:end], G_mean[2:end], yerr=G_err[2:end],
    seriestype=:scatter,xaxis=:log,yaxis=:log, xlabel="p", ylabel="G(p)",
    title="128^2, Broken phase", label="Data", markerstrokecolor=:auto)


plot(k_vals[2:end], (1 ./G_mean[2:end])./(k_vals[2:end].^2),
    seriestype=:scatter,yaxis=:log,xaxis=:log, xlabel="p", ylabel="G(p)",
    title="128^2, Broken phase", label="Data", markerstrokecolor=:auto)

using DelimitedFiles
writedlm("2Dphi4/trainingdata/G_k_128_jk_hmc_tree12.dat", [k_vals G_mean G_err])
