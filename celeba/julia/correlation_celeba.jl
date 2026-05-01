# Correlation analysis for CelebA grayscale diffusion model samples.
# Supports both 64x64 and 128x128 resolutions.
#
# Usage:
#   julia correlation_celeba.jl

using NPZ
using Statistics
using FFTW
using Plots
using DelimitedFiles
using Printf

include("../2Dphi4/CorrelationUtils.jl")

const BASE_DIR = @__DIR__
const MAX_DIAG = 0.51

# ═══════════════════════════════════════════════════════════════════
#  Path helpers
# ═══════════════════════════════════════════════════════════════════

function celeba_model_dir(L::Int, network::String)
    joinpath(BASE_DIR, "celeba_$(L)_$(network)")
end

function train_data_path(L::Int, network::String)
    joinpath(celeba_model_dir(L, network),
             "data", "celeba_train_$(L)x$(L).npy")
end

function dm_sample_path(L::Int, network::String, epoch::String;
                        method::Symbol=:em, subdir::String="data")
    joinpath(celeba_model_dir(L, network),
             subdir, "samples_$(method)_epoch=$(epoch).npy")
end

function correlation_dir(L::Int, network::String)
    dir = joinpath(celeba_model_dir(L, network), "correlation")
    mkpath(dir)
    return dir
end

function discover_epochs(L::Int, network::String;
                         method::Symbol=:em, subdir::String="data")
    dir = joinpath(celeba_model_dir(L, network), subdir)
    isdir(dir) || return String[]
    pat = Regex("^samples_$(method)_epoch=(\\d+)\\.npy\$")
    epochs = String[]
    for f in readdir(dir)
        m = match(pat, f)
        m !== nothing && push!(epochs, m.captures[1])
    end
    return sort(epochs)
end

# ═══════════════════════════════════════════════════════════════════
#  Core
# ═══════════════════════════════════════════════════════════════════

function compute_propagators(;
        L::Int,
        network::String      = "ncsnpp",
        method::Symbol       = :em,
        epoch_list::Union{Vector{String}, Nothing} = nothing,
        data_subdir::String  = "data",
        direction::Symbol    = :radial,
        save::Bool           = true,
        do_plot::Bool        = true,
    )

    dir_label = direction == :radial ? "radial" : string(direction)
    println("\n", "="^60)
    println("  CelebA $(L)x$(L), network=$network, method=$method, direction=$dir_label")
    println("="^60)

    # ── Training data (reference) — use cache if available ──────
    outdir = correlation_dir(L, network)
    dir_suffix = direction == :radial ? "" : "_$(dir_label)"
    cache_file = joinpath(outdir, "G_k_train_celeba_$(L)$(dir_suffix).dat")

    if isfile(cache_file)
        println("  Loading cached training propagator: $cache_file")
        cached = readdlm(cache_file)
        k_train     = cached[:, 1]
        G_train     = cached[:, 2]
        G_train_err = cached[:, 3]
    else
        train_file = train_data_path(L, network)
        if !isfile(train_file)
            @warn "Training data not found" path=train_file
            return nothing
        end
        println("  Computing training propagator (first time, direction=$dir_label)...")
        println("  Training data: $train_file")
        cfgs_train = npzread(train_file)
        k_train, G_train, G_train_err = propagator_radial_bootstrap(cfgs_train;
                                            max_diagonality=MAX_DIAG,
                                            direction=direction)

        if save
            writedlm(cache_file, [k_train G_train G_train_err])
            println("  ✓ Saved training data propagator (cached for future runs)")
        end
    end

    # ── Discover / validate epoch list ───────────────────────────
    if epoch_list === nothing
        epoch_list = discover_epochs(L, network;
                                     method=method, subdir=data_subdir)
    end
    if isempty(epoch_list)
        @warn "No DM sample files found"
        return (k_vals=k_train, G_train=G_train, G_train_err=G_train_err,
                dm_results=Dict{String,Any}())
    end
    println("  Epochs: ", join(epoch_list, ", "))

    # ── Plot ─────────────────────────────────────────────────────
    local p
    if do_plot
        p = plot(k_train[2:end], G_train[2:end]; yerr=G_train_err[2:end],
                 seriestype=:scatter, xaxis=:log,
                 xlabel="|k|", ylabel="G(k)",
                 title="CelebA $(L)×$(L) — $(network)/$(method) ($(dir_label))",
                 label="Training", markerstrokecolor=:auto, legend=:topright)
    end

    dm_results = Dict{String, NamedTuple}()

    for epoch in epoch_list
        fpath = dm_sample_path(L, network, epoch;
                               method=method, subdir=data_subdir)
        if !isfile(fpath)
            @warn "File not found, skipping" path=fpath
            continue
        end
        println("  Processing epoch=$epoch …")
        cfgs_dm = npzread(fpath)
        k_dm, G_dm, G_dm_err = propagator_radial_bootstrap(cfgs_dm;
                                    max_diagonality=MAX_DIAG,
                                    direction=direction)

        dm_results[epoch] = (k_vals=k_dm, G_mean=G_dm, G_err=G_dm_err)

        if save
            outdir = correlation_dir(L, network)
            writedlm(joinpath(outdir,
                "G_k_DM_celeba_$(L)_$(method)$(dir_suffix)_epoch=$(epoch).dat"),
                [k_dm G_dm G_dm_err])
        end

        if do_plot
            plot!(p, k_dm[2:end], G_dm[2:end]; yerr=G_dm_err[2:end],
                  seriestype=:scatter, label="DM-$epoch",
                  markerstrokecolor=:auto)
        end
    end

    if do_plot && save
        outdir = correlation_dir(L, network)
        savefig(p, joinpath(outdir,
            "propagator_celeba_$(L)_$(method)$(dir_suffix).pdf"))
        println("  ✓ Saved plot")
    end

    return (k_vals=k_train, G_train=G_train, G_train_err=G_train_err,
            dm_results=dm_results)
end

# ═══════════════════════════════════════════════════════════════════
#  Merged-epoch propagator: average over a window of epochs
# ═══════════════════════════════════════════════════════════════════

"""
    compute_merged_propagator(; L, network, epoch_list, label, ...)

Load DM samples from multiple epochs, concatenate them, and compute
a single propagator. This averages out parameter oscillations.
"""
function compute_merged_propagator(;
        L::Int,
        network::String,
        epoch_list::Vector{String},
        method::Symbol           = :em,
        label::String            = "merged",
        data_subdir::String      = "data",
        direction::Symbol        = :radial,
        save::Bool               = true,
    )
    dir_label = direction == :radial ? "radial" : string(direction)
    dir_suffix = direction == :radial ? "" : "_$(dir_label)"

    println("\n  Merging epochs [", join(epoch_list, ","), "] -> $label")

    all_cfgs = []
    for epoch in epoch_list
        fpath = dm_sample_path(L, network, epoch;
                               method=method, subdir=data_subdir)
        isfile(fpath) || continue
        cfgs = npzread(fpath)
        push!(all_cfgs, cfgs)
    end

    isempty(all_cfgs) && (@warn "No files found"; return nothing)

    merged = cat(all_cfgs...; dims=ndims(all_cfgs[1]))
    N_total = size(merged, ndims(merged))
    println("    $N_total samples merged")

    k_vals, G_mean, G_err = propagator_radial_bootstrap(merged;
                                max_diagonality=MAX_DIAG,
                                direction=direction)

    if save
        outdir = correlation_dir(L, network)
        writedlm(joinpath(outdir,
            "G_k_DM_celeba_$(L)_$(method)$(dir_suffix)_$(label).dat"),
            [k_vals G_mean G_err])
    end

    return (k_vals=k_vals, G_mean=G_mean, G_err=G_err)
end

# ═══════════════════════════════════════════════════════════════════
#  Sliding-window propagator evolution
# ═══════════════════════════════════════════════════════════════════

"""
    compute_sliding_window(; L, network, window_radius, direction, ...)

For each available epoch, merge samples from neighboring epochs within
±window_radius, compute propagator. This gives a smooth G(k) vs epoch curve.

Example:
    compute_sliding_window(L=128, network="ncsnpp", window_radius=2)
    # epoch 0199: merges [0099, 0199, 0299] if available
"""
function compute_sliding_window(;
        L::Int,
        network::String      = "ncsnpp",
        method::Symbol       = :em,
        data_subdir::String  = "data",
        direction::Symbol    = :radial,
        window_radius::Int   = 2,
        save::Bool           = true,
        do_plot::Bool        = true,
    )
    dir_label = direction == :radial ? "radial" : string(direction)
    dir_suffix = direction == :radial ? "" : "_$(dir_label)"

    println("\n", "="^60)
    println("  Sliding window: CelebA $(L)×$(L), network=$network, method=$method")
    println("  window_radius=$window_radius, direction=$dir_label")
    println("="^60)

    # Load training reference (cached)
    outdir = correlation_dir(L, network)
    cache_file = joinpath(outdir, "G_k_train_celeba_$(L)$(dir_suffix).dat")
    if isfile(cache_file)
        cached = readdlm(cache_file)
        k_train, G_train, G_train_err = cached[:,1], cached[:,2], cached[:,3]
    else
        train_file = train_data_path(L, network)
        isfile(train_file) || (@warn "No training data"; return nothing)
        println("  Computing training propagator...")
        cfgs_train = npzread(train_file)
        k_train, G_train, G_train_err = propagator_radial_bootstrap(cfgs_train;
                                            max_diagonality=MAX_DIAG, direction=direction)
        save && writedlm(cache_file, [k_train G_train G_train_err])
    end

    # Discover all epochs and sort numerically
    all_epochs = discover_epochs(L, network; method=method, subdir=data_subdir)
    isempty(all_epochs) && (@warn "No sample files"; return nothing)
    epoch_nums = parse.(Int, all_epochs)
    sorted_idx = sortperm(epoch_nums)
    all_epochs = all_epochs[sorted_idx]
    epoch_nums = epoch_nums[sorted_idx]
    println("  Available epochs: $(length(all_epochs))")

    # For each epoch, merge with neighbors within window
    results = Dict{Int, NamedTuple}()

    for (i, epoch) in enumerate(all_epochs)
        lo = max(1, i - window_radius)
        hi = min(length(all_epochs), i + window_radius)
        window_epochs = all_epochs[lo:hi]

        all_cfgs = []
        for we in window_epochs
            fpath = dm_sample_path(L, network, we;
                                   method=method, subdir=data_subdir)
            isfile(fpath) || continue
            push!(all_cfgs, npzread(fpath))
        end
        isempty(all_cfgs) && continue

        merged = cat(all_cfgs...; dims=ndims(all_cfgs[1]))
        N_total = size(merged, ndims(merged))

        k_dm, G_dm, G_dm_err = propagator_radial_bootstrap(merged;
                                    max_diagonality=MAX_DIAG, direction=direction)

        ep_num = epoch_nums[i]
        results[ep_num] = (k_vals=k_dm, G_mean=G_dm, G_err=G_dm_err, N=N_total)

        if save
            writedlm(joinpath(outdir,
                "G_k_DM_celeba_$(L)_$(method)$(dir_suffix)_sw$(window_radius)_epoch=$(epoch).dat"),
                [k_dm G_dm G_dm_err])
        end
        println("  epoch=$(epoch): window=[$(all_epochs[lo])..$(all_epochs[hi])], N=$N_total")
    end

    # Plot: pick representative k-bins and show G(k) vs epoch
    if do_plot && !isempty(results)
        sorted_eps = sort(collect(keys(results)))
        # Lightning's `epoch=NNNN.ckpt` is saved at the END of training epoch N;
        # the +1 shift converts to "epochs completed" and keeps epoch=0000
        # visible on a log x-axis.
        plot_epochs = sorted_eps .+ 1
        nb = length(first(values(results)).k_vals)

        # Skip index 1 (k=0): G(k=0)=0 by construction (subtract_mean=true)
        k_indices = unique(clamp.([2, 3, 4, nb÷4, nb÷2, nb-1], 2, nb))

        p = plot(xlabel="Epochs completed", ylabel="G(k) / G_train(k)",
                 title="CelebA $(L)×$(L) — Propagator evolution ($(method), $(dir_label), window=$window_radius)",
                 legend=:topright)

        for ki in k_indices
            k_val = first(values(results)).k_vals[ki]
            G_ref = G_train[ki]

            ratios = [results[ep].G_mean[ki] / G_ref for ep in sorted_eps]
            errs   = [results[ep].G_err[ki] / G_ref for ep in sorted_eps]

            plot!(p, plot_epochs, ratios; yerr=errs,
                  label=@sprintf("|k|=%.3f", k_val),
                  markerstrokecolor=:auto, seriestype=:scatter, xaxis=:log)
        end
        hline!(p, [1.0]; linestyle=:dash, color=:gray, label="")

        if save
            savefig(p, joinpath(outdir,
                "propagator_evolution_celeba_$(L)_$(method)$(dir_suffix)_sw$(window_radius).pdf"))
            println("  ✓ Saved evolution plot")
        end
    end

    return results
end

# ═══════════════════════════════════════════════════════════════════
#  Parameter sets
# ═══════════════════════════════════════════════════════════════════

directions = [:radial, :x, :y]

param_sets = vec([
    (L=L, network="ncsnpp", method=method,
     epoch_list=nothing, data_subdir="data", direction=dir)
    for L in (64, 128), method in (:em, :dpm2), dir in directions
])

for ps in param_sets
    compute_propagators(;
        L           = ps.L,
        network     = ps.network,
        method      = ps.method,
        epoch_list  = ps.epoch_list,
        data_subdir = ps.data_subdir,
        direction   = ps.direction,
    )
end

# Sliding-window propagator evolution (G(k)/G_train vs epoch per k-bin)
for L in (64, 128), method in (:em, :dpm2), dir in directions
    compute_sliding_window(L=L, network="ncsnpp", method=method,
                           window_radius=2, direction=dir)
end
