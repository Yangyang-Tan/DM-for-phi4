# Correlation analysis for STL-10 grayscale diffusion model samples (64x64).
# Computes radially-averaged momentum propagator with bootstrap errors,
# comparing training data (reference) against DM samples at each epoch.
#
# Usage:
#   julia correlation_stl10.jl
#   (edit param_sets at the bottom)

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

function stl10_model_dir(output_name::String, network::String)
    joinpath(BASE_DIR, "stl10_$(output_name)_$(network)")
end

function train_data_path(output_name::String, network::String, class_name::String)
    joinpath(stl10_model_dir(output_name, network),
             "data", "stl10_$(class_name)_train_64x64.npy")
end

function dm_sample_path(output_name::String, network::String, epoch::String;
                        subdir::String="data")
    joinpath(stl10_model_dir(output_name, network),
             subdir, "samples_epoch=epoch=$(epoch).npy")
end

function correlation_dir(output_name::String, network::String)
    dir = joinpath(stl10_model_dir(output_name, network), "correlation")
    mkpath(dir)
    return dir
end

function discover_epochs(output_name::String, network::String;
                         subdir::String="data")
    dir = joinpath(stl10_model_dir(output_name, network), subdir)
    isdir(dir) || return String[]
    epochs = String[]
    for f in readdir(dir)
        m = match(r"^samples_epoch=epoch=(\d+)\.npy$", f)
        m !== nothing && push!(epochs, m.captures[1])
    end
    return sort(epochs)
end

# ═══════════════════════════════════════════════════════════════════
#  Core
# ═══════════════════════════════════════════════════════════════════

"""
    compute_propagators(; output_name, class_name, network, ...)

Compute propagators for training data (reference) and DM samples.
"""
function compute_propagators(;
        output_name::String,
        class_name::String   = "all",
        network::String      = "ncsnpp",
        epoch_list::Union{Vector{String}, Nothing} = nothing,
        data_subdir::String  = "data",
        direction::Symbol    = :radial,
        save::Bool           = true,
        do_plot::Bool        = true,
    )

    dir_label = direction == :radial ? "radial" : string(direction)
    println("\n", "="^60)
    println("  STL-10: output_name=$output_name, class=$class_name, network=$network, direction=$dir_label")
    println("="^60)

    # ── Training data (reference) — use cache if available ──────
    outdir = correlation_dir(output_name, network)
    dir_suffix = direction == :radial ? "" : "_$(dir_label)"
    cache_file = joinpath(outdir, "G_k_train_$(class_name)$(dir_suffix).dat")

    if isfile(cache_file)
        println("  Loading cached training propagator: $cache_file")
        cached = readdlm(cache_file)
        k_train     = cached[:, 1]
        G_train     = cached[:, 2]
        G_train_err = cached[:, 3]
    else
        train_file = train_data_path(output_name, network, class_name)
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
        epoch_list = discover_epochs(output_name, network; subdir=data_subdir)
    end
    if isempty(epoch_list)
        @warn "No DM sample files found"
        return (k_vals=k_train, G_train=G_train, G_train_err=G_train_err,
                dm_results=Dict{String,Any}())
    end
    println("  Epochs: ", join(epoch_list, ", "))

    # ── Plot setup ───────────────────────────────────────────────
    local p
    if do_plot
        p = plot(k_train[2:end], G_train[2:end]; yerr=G_train_err[2:end],
                 seriestype=:scatter, xaxis=:log,
                 xlabel="|k|", ylabel="G(k)",
                 title="STL-10 $(class_name) — $(network) (64×64, $(dir_label))",
                 label="Training", markerstrokecolor=:auto, legend=:topright)
    end

    # ── Loop over epochs ─────────────────────────────────────────
    dm_results = Dict{String, NamedTuple}()

    for epoch in epoch_list
        fpath = dm_sample_path(output_name, network, epoch; subdir=data_subdir)
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
            outdir = correlation_dir(output_name, network)
            writedlm(joinpath(outdir,
                        "G_k_DM_$(class_name)$(dir_suffix)_epoch=$(epoch).dat"),
                     [k_dm G_dm G_dm_err])
        end

        if do_plot
            plot!(p, k_dm[2:end], G_dm[2:end]; yerr=G_dm_err[2:end],
                  seriestype=:scatter, label="DM-$epoch",
                  markerstrokecolor=:auto)
        end
    end

    if do_plot
        display(p)
        if save
            outdir = correlation_dir(output_name, network)
            savefig(p, joinpath(outdir, "propagator_$(class_name)$(dir_suffix).pdf"))
            println("  ✓ Saved plot")
        end
    end

    return (k_vals=k_train, G_train=G_train, G_train_err=G_train_err,
            dm_results=dm_results)
end

# ═══════════════════════════════════════════════════════════════════
#  Merged-epoch propagator: average over a window of epochs
# ═══════════════════════════════════════════════════════════════════

"""
    compute_merged_propagator(; output_name, class_name, network, epoch_list, label, ...)

Load DM samples from multiple epochs, concatenate, and compute a single
propagator. Averages out parameter oscillations in the training plateau.
"""
function compute_merged_propagator(;
        output_name::String,
        class_name::String       = "all",
        network::String,
        epoch_list::Vector{String},
        label::String            = "merged",
        data_subdir::String      = "data",
        direction::Symbol        = :radial,
        save::Bool               = true,
    )
    dir_label = direction == :radial ? "radial" : string(direction)
    dir_suffix = direction == :radial ? "" : "_$(dir_label)"

    println("\n", "="^60)
    println("  Merged propagator: label=$label, epochs=", join(epoch_list, ","))
    println("  STL-10 $(class_name), network=$network, direction=$dir_label")
    println("="^60)

    all_cfgs = []
    for epoch in epoch_list
        fpath = dm_sample_path(output_name, network, epoch; subdir=data_subdir)
        if !isfile(fpath)
            @warn "File not found, skipping" path=fpath
            continue
        end
        cfgs = npzread(fpath)
        println("  Loaded epoch=$epoch: $(size(cfgs, ndims(cfgs))) samples")
        push!(all_cfgs, cfgs)
    end

    if isempty(all_cfgs)
        @warn "No sample files found for merging"
        return nothing
    end

    merged = cat(all_cfgs...; dims=ndims(all_cfgs[1]))
    N_total = size(merged, ndims(merged))
    println("  Total merged samples: $N_total")

    k_vals, G_mean, G_err = propagator_radial_bootstrap(merged;
                                max_diagonality=MAX_DIAG,
                                direction=direction)

    if save
        outdir = correlation_dir(output_name, network)
        writedlm(joinpath(outdir, "G_k_DM_$(class_name)$(dir_suffix)_$(label).dat"),
                 [k_vals G_mean G_err])
        println("  ✓ Saved merged propagator: $label")
    end

    return (k_vals=k_vals, G_mean=G_mean, G_err=G_err)
end

# ═══════════════════════════════════════════════════════════════════
#  Parameter sets
# ═══════════════════════════════════════════════════════════════════

param_sets = [
    (output_name="unlabeled_all", class_name="all", network="ncsnpp",
     epoch_list=nothing, data_subdir="data", direction=:radial),
    # (output_name="unlabeled_all", class_name="all", network="ncsnpp",
    #  epoch_list=nothing, data_subdir="data", direction=:x),
    # (output_name="unlabeled_all", class_name="all", network="ncsnpp",
    #  epoch_list=nothing, data_subdir="data", direction=:y),
    # (output_name="unlabeled_all", class_name="all", network="ncsnpp",
    #  epoch_list=nothing, data_subdir="data", direction=:diagonal),
]

for ps in param_sets
    compute_propagators(;
        output_name = ps.output_name,
        class_name  = ps.class_name,
        network     = ps.network,
        epoch_list  = ps.epoch_list,
        data_subdir = ps.data_subdir,
        direction   = ps.direction,
    )
end
