# CorrelationUtils.jl
# Dimension-generic lattice propagator utilities with bootstrap error estimation.
# Supports 2D, 3D lattices and images — any array whose last axis is the sample index.

using FFTW
using Statistics
using Bootstrap
using Random

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Lattice momentum                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    lattice_k_sq(ns, L::Int)

Lattice momentum squared  k̂² = Σᵢ 4 sin²(π nᵢ / L)  for mode indices `ns`.
"""
function lattice_k_sq(ns, L::Int)
    s = 0.0
    for n in ns
        k = 2π * n / L
        s += 4sin(k / 2)^2
    end
    return s
end

"""
    diagonality(ns, L::Int)

Diagonality parameter  Σᵢ pᵢ⁴ / (Σᵢ pᵢ²)².
Low values ≈ 1/D (diagonal); high values ≈ 1 (axial).
Returns 0.0 for the zero mode.
"""
function diagonality(ns, L::Int)
    sum_p2 = 0.0
    sum_p4 = 0.0
    for n in ns
        k = 2π * n / L
        p2 = 4sin(k / 2)^2
        sum_p2 += p2
        sum_p4 += p2^2
    end
    sum_p2 < 1e-10 && return 0.0
    return sum_p4 / sum_p2^2
end

"""
    lattice_k_sq_array(L::Int, D::Int)

Return a D-dimensional array of k̂² values for a cubic lattice of side `L`.
"""
function lattice_k_sq_array(L::Int, D::Int)
    shape = ntuple(_ -> L, D)
    kh2 = zeros(Float64, shape)
    for idx in CartesianIndices(shape)
        ns = ntuple(d -> begin
            n = idx[d] - 1
            n > L ÷ 2 ? n - L : n
        end, D)
        kh2[idx] = lattice_k_sq(ns, L)
    end
    return kh2
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Momentum binning (precomputed, reusable across configs)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

struct MomentumBins{D}
    k_vals::Vector{Float64}
    bin_indices::Vector{Vector{CartesianIndex{D}}}
end

"""
    compute_momentum_bins(spatial_size; max_diagonality=0.51)

Precompute radial momentum bins for a lattice of given `spatial_size`.
Assumes a cubic lattice (L = spatial_size[1]).
Modes with diagonality > `max_diagonality` are filtered out.
"""
function compute_momentum_bins(spatial_size::NTuple{D,Int};
        max_diagonality::Float64=0.51) where D
    L = spatial_size[1]
    k_sq_list  = Float64[]
    idx_list   = CartesianIndex{D}[]

    for idx in CartesianIndices(spatial_size)
        ns = ntuple(d -> begin
            n = idx[d] - 1
            n > L ÷ 2 ? n - L : n
        end, D)

        diag = diagonality(ns, L)
        diag > max_diagonality && continue

        push!(k_sq_list, lattice_k_sq(ns, L))
        push!(idx_list, idx)
    end

    unique_k_sq = sort(unique(round.(k_sq_list, digits=5)))
    k_vals      = sqrt.(unique_k_sq)

    bin_indices = Vector{Vector{CartesianIndex{D}}}(undef, length(unique_k_sq))
    for (j, ksq) in enumerate(unique_k_sq)
        mask = abs.(k_sq_list .- ksq) .< 1e-4
        bin_indices[j] = idx_list[mask]
    end
    return MomentumBins{D}(k_vals, bin_indices)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Radial average                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    radial_average(G_k, bins::MomentumBins)

Radial average of a momentum-space quantity using precomputed `bins`.
Returns a `Vector{Float64}` of length `length(bins.k_vals)`.
"""
function radial_average(G_k::AbstractArray, bins::MomentumBins)
    nb = length(bins.k_vals)
    G_avg = Vector{Float64}(undef, nb)
    for j in 1:nb
        s = 0.0
        for idx in bins.bin_indices[j]
            s += G_k[idx]
        end
        G_avg[j] = s / length(bins.bin_indices[j])
    end
    return G_avg
end

"""
    radial_average(G_k; max_diagonality=0.51)

Convenience: compute bins on the fly and return `(k_vals, G_avg)`.
"""
function radial_average(G_k::AbstractArray{T}; max_diagonality::Float64=0.51) where T
    bins = compute_momentum_bins(size(G_k); max_diagonality=max_diagonality)
    return bins.k_vals, radial_average(G_k, bins)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Propagator computation                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    momentum_propagator(cfgs::AbstractArray; subtract_mean=true)

Mean momentum-space propagator  G(k) = ⟨|φ̃(k)|²⟩ / V.
`cfgs` has shape `(L₁, …, L_D, N_configs)`.  Last axis = sample index.
"""
function momentum_propagator(cfgs::AbstractArray; subtract_mean::Bool=true)
    nd = ndims(cfgs)
    D  = nd - 1
    spatial = size(cfgs)[1:D]
    V  = prod(spatial)
    N  = size(cfgs, nd)

    G_sum = zeros(Float64, spatial)
    for i in 1:N
        cfg = collect(selectdim(cfgs, nd, i))
        if subtract_mean
            cfg .-= mean(cfg)
        end
        phi_k = fft(cfg) ./ sqrt(V)
        G_sum .+= abs2.(phi_k)
    end
    return G_sum ./ N
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Bootstrap error estimation                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    propagator_radial_bootstrap(cfgs::AbstractArray;
        n_boot=1000, subtract_mean=true, max_diagonality=0.51, seed=nothing)

Radially-averaged momentum propagator with bootstrap error bars.
Returns `(k_vals, G_mean, G_err)`.

Steps:
1. For each config, compute |φ̃(k)|²/V and radially average → per-config, per-bin values.
2. Bootstrap the mean of each radial bin using Bootstrap.jl.
"""
function propagator_radial_bootstrap(cfgs::AbstractArray{T};
        n_boot::Int            = 1000,
        subtract_mean::Bool    = true,
        max_diagonality::Float64 = 0.51,
        seed::Union{Int,Nothing} = nothing,
    ) where T

    nd = ndims(cfgs)
    D  = nd - 1
    spatial = size(cfgs)[1:D]
    V  = prod(spatial)
    N  = size(cfgs, nd)

    bins = compute_momentum_bins(spatial; max_diagonality=max_diagonality)
    nb   = length(bins.k_vals)

    G_rad = zeros(Float64, nb, N)

    for i in 1:N
        cfg = collect(selectdim(cfgs, nd, i))
        if subtract_mean
            cfg .-= mean(cfg)
        end
        phi_k = fft(cfg) ./ sqrt(V)
        G_rad[:, i] .= radial_average(abs2.(phi_k), bins)
    end

    if seed !== nothing
        Random.seed!(seed)
    end

    G_mean = zeros(Float64, nb)
    G_err  = zeros(Float64, nb)

    for j in 1:nb
        bs = bootstrap(mean, G_rad[j, :], BasicSampling(n_boot))
        G_mean[j] = bs.t0[1]
        G_err[j]  = stderror(bs)[1]
    end

    return bins.k_vals, G_mean, G_err
end

"""
    momentum_propagator_bootstrap(cfgs::AbstractArray;
        n_boot=200, subtract_mean=true, seed=nothing)

Full (non-radially-averaged) momentum propagator with bootstrap errors.
Returns `(G_mean, G_err)` arrays of the same spatial shape as a single config.

Uses a streaming approach: per-config G(k) values are accumulated into
bootstrap sums without storing the full `(spatial..., N)` array.
"""
function momentum_propagator_bootstrap(cfgs::AbstractArray{T};
        n_boot::Int            = 200,
        subtract_mean::Bool    = true,
        seed::Union{Int,Nothing} = nothing,
    ) where T

    nd = ndims(cfgs)
    D  = nd - 1
    spatial = size(cfgs)[1:D]
    V  = prod(spatial)
    N  = size(cfgs, nd)

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    boot_counts = zeros(Int, N, n_boot)
    for b in 1:n_boot
        for _ in 1:N
            boot_counts[rand(rng, 1:N), b] += 1
        end
    end

    G_sum      = zeros(Float64, spatial)
    boot_sums  = zeros(Float64, spatial..., n_boot)

    for i in 1:N
        cfg = collect(selectdim(cfgs, nd, i))
        if subtract_mean
            cfg .-= mean(cfg)
        end
        phi_k = fft(cfg) ./ sqrt(V)
        G_k_i = abs2.(phi_k)

        G_sum .+= G_k_i

        for b in 1:n_boot
            c = boot_counts[i, b]
            c == 0 && continue
            selectdim(boot_sums, D + 1, b) .+= c .* G_k_i
        end
    end

    G_mean = G_sum ./ N
    for b in 1:n_boot
        selectdim(boot_sums, D + 1, b) ./= N
    end
    G_err = dropdims(std(boot_sums; dims=D + 1); dims=D + 1)

    return G_mean, G_err
end
