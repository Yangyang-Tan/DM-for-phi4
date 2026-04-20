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
    compute_momentum_bins(spatial_size; max_diagonality=0.51, direction=:radial)

Precompute momentum bins for a lattice of given `spatial_size`.
Assumes a cubic lattice (L = spatial_size[1]).

`direction` controls which momentum modes are selected:
- `:radial`   — all modes with diagonality ≤ `max_diagonality`, radially averaged (default)
- `:x`        — axial modes along x: (nx, 0, …), k = |k̂_x|
- `:y`        — axial modes along y: (0, ny, …), k = |k̂_y|
- `:diagonal` — diagonal modes: (n, n, …),       k = √(D·k̂²(n))
"""
function compute_momentum_bins(spatial_size::NTuple{D,Int};
        max_diagonality::Float64=0.51,
        direction::Symbol=:radial) where D
    L = spatial_size[1]
    k_sq_list  = Float64[]
    idx_list   = CartesianIndex{D}[]

    for idx in CartesianIndices(spatial_size)
        ns = ntuple(d -> begin
            n = idx[d] - 1
            n > L ÷ 2 ? n - L : n
        end, D)

        if direction == :x
            # Only modes along x-axis: all other components = 0
            all(ns[d] == 0 for d in 2:D) || continue
        elseif direction == :y
            # Only modes along y-axis: all other components = 0
            D >= 2 || error("direction=:y requires D ≥ 2")
            ns[1] == 0 || continue
            all(ns[d] == 0 for d in 3:D) || continue
        elseif direction == :diagonal
            # Only diagonal modes: all components equal
            all(ns[d] == ns[1] for d in 2:D) || continue
        else  # :radial
            diag = diagonality(ns, L)
            diag > max_diagonality && continue
        end

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
    radial_average(G_k; max_diagonality=0.51, direction=:radial)

Convenience: compute bins on the fly and return `(k_vals, G_avg)`.
"""
function radial_average(G_k::AbstractArray{T};
        max_diagonality::Float64=0.51,
        direction::Symbol=:radial) where T
    bins = compute_momentum_bins(size(G_k); max_diagonality=max_diagonality,
                                 direction=direction)
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
        n_boot=1000, subtract_mean=true, max_diagonality=0.51,
        direction=:radial, seed=nothing)

Momentum propagator with bootstrap error bars.
Returns `(k_vals, G_mean, G_err)`.

`direction`: `:radial` (default), `:x`, `:y`, or `:diagonal`.
See `compute_momentum_bins` for details.
"""
function propagator_radial_bootstrap(cfgs::AbstractArray{T};
        n_boot::Int            = 1000,
        subtract_mean::Bool    = true,
        max_diagonality::Float64 = 0.51,
        direction::Symbol      = :radial,
        seed::Union{Int,Nothing} = nothing,
    ) where T

    nd = ndims(cfgs)
    D  = nd - 1
    spatial = size(cfgs)[1:D]
    V  = prod(spatial)
    N  = size(cfgs, nd)

    bins = compute_momentum_bins(spatial; max_diagonality=max_diagonality,
                                 direction=direction)
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

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Propagator-mismatch diagnostics (Gaussian-level)                        ║
# ║                                                                           ║
# ║  Given radially-averaged propagators G_train(k), G_dm(k) and their        ║
# ║  bootstrap errors, compute three complementary diagnostics:               ║
# ║    (1) per-mode Gaussian KL    D_k = ½(r - 1 - log r), r = G_t / G_dm    ║
# ║    (2) statistical z-score     z_k = (G_dm - G_t) / √(σ_t² + σ_dm²)     ║
# ║    (3) phase-space weighted Δ   w_k = k^(D-1) (G_dm - G_t)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    per_mode_kl(G_dm, G_train; G_dm_err=nothing, G_train_err=nothing, eps=1e-30)

Per-mode Gaussian KL  D_k = ½ (r - 1 - log r)  with  r = G_train / G_dm,
which equals the KL divergence D_KL(p_train ‖ p_dm) between two zero-mean
Gaussian fields that share the propagators as covariances (diagonal in
momentum due to translation invariance).

If bootstrap errors are supplied, a linearized 1σ uncertainty on D_k is
also returned via Gaussian error propagation:
    ∂D/∂G_t  = (1/G_dm - 1/G_t) / 2
    ∂D/∂G_dm = (1/G_dm - G_t/G_dm²) / 2
Near perfect match (r≈1) the linearization degrades — a noise-floor
estimator  D_floor ≈ (σ_t² + σ_dm²) / (4 G²)  is also returned so the
user can judge statistical significance.

Returns `(D_k, D_k_err, D_floor)` (`D_k_err` and `D_floor` are `nothing`
if errors were not supplied).
"""
function per_mode_kl(G_dm::AbstractVector, G_train::AbstractVector;
        G_dm_err::Union{AbstractVector,Nothing}    = nothing,
        G_train_err::Union{AbstractVector,Nothing} = nothing,
        eps::Float64 = 1e-30,
    )
    @assert length(G_dm) == length(G_train)
    Gd = max.(G_dm,    eps)
    Gt = max.(G_train, eps)
    r  = Gt ./ Gd
    D  = 0.5 .* (r .- 1 .- log.(r))

    if G_dm_err === nothing || G_train_err === nothing
        return D, nothing, nothing
    end

    dD_dGt  = 0.5 .* (1 ./ Gd .- 1 ./ Gt)
    dD_dGd  = 0.5 .* (1 ./ Gd .- Gt ./ Gd.^2)
    D_err   = sqrt.(dD_dGt.^2 .* G_train_err.^2 .+ dD_dGd.^2 .* G_dm_err.^2)

    Gmean   = 0.5 .* (Gd .+ Gt)
    D_floor = (G_train_err.^2 .+ G_dm_err.^2) ./ (4 .* Gmean.^2)

    return D, D_err, D_floor
end

"""
    propagator_zscore(G_dm, G_train, G_dm_err, G_train_err)

Statistical z-score per k-bin,
    z_k = (G_dm(k) - G_train(k)) / √(σ_dm(k)² + σ_train(k)²),
measuring whether the DM/training propagator discrepancy is significant
compared to the bootstrap uncertainties. |z_k| ≲ 2 ⇒ consistent at 2σ.
"""
function propagator_zscore(G_dm::AbstractVector, G_train::AbstractVector,
                           G_dm_err::AbstractVector, G_train_err::AbstractVector)
    @assert length(G_dm) == length(G_train) == length(G_dm_err) == length(G_train_err)
    denom = sqrt.(G_dm_err.^2 .+ G_train_err.^2)
    denom = max.(denom, 1e-30)
    return (G_dm .- G_train) ./ denom
end

"""
    phase_space_weighted_delta(k_vals, G_dm, G_train; D::Int,
                               G_dm_err=nothing, G_train_err=nothing)

Phase-space weighted propagator difference
    w_k = k^(D-1) · (G_dm(k) - G_train(k))
so that  ∫ dk · w_k  is proportional to the mismatch in the spatial-variance
contribution ⟨φ²⟩ = ∫ d^Dk/(2π)^D · G(k). The prefactor k^(D-1) accounts
for the radial phase space of a D-dimensional momentum integral, making
UV and IR contributions to the total field variance directly comparable.

Returns `(w_k, w_k_err)`; `w_k_err` is `nothing` if errors are not provided.
"""
function phase_space_weighted_delta(k_vals::AbstractVector,
                                    G_dm::AbstractVector,
                                    G_train::AbstractVector;
                                    D::Int,
                                    G_dm_err::Union{AbstractVector,Nothing}    = nothing,
                                    G_train_err::Union{AbstractVector,Nothing} = nothing,
    )
    @assert length(k_vals) == length(G_dm) == length(G_train)
    weight = k_vals.^(D - 1)
    w      = weight .* (G_dm .- G_train)
    w_err  = (G_dm_err === nothing || G_train_err === nothing) ? nothing :
             weight .* sqrt.(G_dm_err.^2 .+ G_train_err.^2)
    return w, w_err
end
