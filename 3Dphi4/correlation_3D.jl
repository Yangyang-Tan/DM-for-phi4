using JLD2
using Statistics
using NPZ
using Plots
using FFTW
using DelimitedFiles

include(joinpath(@__DIR__, "..", "2Dphi4", "CorrelationUtils.jl"))

const BASE_DIR = @__DIR__
const LAMBDA = 0.9
const MAX_DIAG = 0.34   # 3D: most-diagonal mode (1,1,1) has diagonality 1/3 ≈ 0.333

# ═══════════════════════════════════════════════════════════════════
#  Path helpers  (3D naming: phi4_3d_L{L}_k{k}_l{l}_{network})
# ═══════════════════════════════════════════════════════════════════

function hmc_data_path(L::Int, k::Real; l::Real=LAMBDA)
    joinpath(BASE_DIR, "trainingdata",
             "cfgs_wolff_fahmc_k=$(k)_l=$(l)_$(L)^3.jld2")
end

function dm_model_dir(L::Int, k::Real; network::String="ncsnpp", l::Real=LAMBDA)
    suffix = network == "" ? "" : "_$(network)"
    joinpath(BASE_DIR, "phi4_3d_L$(L)_k$(k)_l$(l)$(suffix)")
end

function dm_sample_path(L::Int, k::Real, epoch::String;
                        network::String="ncsnpp", l::Real=LAMBDA,
                        subdir::String="data")
    joinpath(dm_model_dir(L, k; network=network, l=l),
             subdir, "samples_epoch=$(epoch).npy")
end

"""Short kappa label for output filenames: 0.2 → \"2\", 0.18 → \"18\"."""
kappa_label(k::Real) = replace(string(k), "0." => "")

function correlation_dir(L::Int, k::Real;
                         network::String="ncsnpp", l::Real=LAMBDA,
                         tag::String="")
    dir = joinpath(dm_model_dir(L, k; network=network, l=l),
                   "correlation" * tag)
    mkpath(dir)
    return dir
end

"""
Auto-discover all epoch strings available under the sample directory.
Returns a sorted vector like ["0009", "0049", "0099", ...].
"""
function discover_epochs(L::Int, k::Real;
                         network::String="ncsnpp", l::Real=LAMBDA,
                         subdir::String="data")
    dir = joinpath(dm_model_dir(L, k; network=network, l=l), subdir)
    isdir(dir) || return String[]
    epochs = String[]
    for f in readdir(dir)
        m = match(r"^samples_epoch=(\d+)\.npy$", f)
        m !== nothing && push!(epochs, m.captures[1])
    end
    return sort(epochs)
end

# ═══════════════════════════════════════════════════════════════════
#  Core: compute propagators for one (L, κ) parameter set
# ═══════════════════════════════════════════════════════════════════

"""
    compute_propagators(; L, k, ...)

Compute radially-averaged momentum propagators with bootstrap errors
for HMC reference data and DM samples at each training epoch.

Keyword arguments
─────────────────
- `L`            : lattice size (L × L × L)
- `k`            : hopping parameter κ
- `l`            : coupling constant λ  (default $(LAMBDA))
- `network`      : model architecture suffix  (default "ncsnpp")
- `epoch_list`   : explicit list of epoch strings, or `nothing` to auto-discover
- `data_subdir`  : subdirectory under model dir containing .npy files (default "data")
- `corr_tag`     : extra tag appended to output "correlation" dir (e.g. "_ema")
- `save`         : write .dat files  (default true)
- `do_plot`      : create comparison plot  (default true)
"""
function compute_propagators(;
        L::Int,
        k::Real,
        l::Real              = LAMBDA,
        network::String      = "ncsnpp",
        epoch_list::Union{Vector{String}, Nothing} = nothing,
        data_subdir::String  = "data",
        corr_tag::String     = "",
        save::Bool           = true,
        do_plot::Bool        = true,
    )

    kl = kappa_label(k)
    println("\n", "="^60)
    println("  3D φ⁴:  L = $L,  κ = $k,  λ = $l,  network = $network")
    println("="^60)

    # ── HMC reference ────────────────────────────────────────────
    hmc_file = hmc_data_path(L, k; l=l)
    if !isfile(hmc_file)
        @warn "HMC data not found, skipping" path=hmc_file
        return nothing
    end
    println("  HMC data: $hmc_file")
    cfgs_hmc = load(hmc_file)["cfgs"]
    k_hmc, G_hmc, G_hmc_err = propagator_radial_bootstrap(cfgs_hmc;
                                   max_diagonality=MAX_DIAG)

    if save
        outdir = correlation_dir(L, k; network=network, l=l, tag=corr_tag)
        writedlm(joinpath(outdir, "G_k_$(L)_boot_HMC_$(kl)_$(l).dat"),
                 [k_hmc G_hmc G_hmc_err])
        println("  ✓ Saved HMC propagator")
    end

    # ── Discover / validate epoch list ───────────────────────────
    if epoch_list === nothing
        epoch_list = discover_epochs(L, k; network=network, l=l,
                                     subdir=data_subdir)
    end
    if isempty(epoch_list)
        @warn "No DM sample files found"
        return (k_vals=k_hmc, G_hmc=G_hmc, G_hmc_err=G_hmc_err,
                dm_results=Dict{String,Any}())
    end
    println("  Epochs: ", join(epoch_list, ", "))

    # ── Plot setup ───────────────────────────────────────────────
    local p
    if do_plot
        p = plot(k_hmc[2:end], G_hmc[2:end]; yerr=G_hmc_err[2:end],
                 seriestype=:scatter, xaxis=:log,
                 xlabel="p", ylabel="G(p)",
                 title="3D  L=$(L), κ=$(k)",
                 label="HMC", markerstrokecolor=:auto, legend=:topright)
    end

    # ── Loop over epochs ─────────────────────────────────────────
    dm_results = Dict{String, NamedTuple}()

    for epoch in epoch_list
        fpath = dm_sample_path(L, k, epoch;
                    network=network, l=l, subdir=data_subdir)
        if !isfile(fpath)
            @warn "File not found, skipping" path=fpath
            continue
        end
        println("  Processing epoch=$epoch …")
        cfgs_dm = npzread(fpath)
        k_dm, G_dm, G_dm_err = propagator_radial_bootstrap(cfgs_dm;
                                    max_diagonality=MAX_DIAG)

        dm_results[epoch] = (k_vals=k_dm, G_mean=G_dm, G_err=G_dm_err)

        if save
            outdir = correlation_dir(L, k; network=network, l=l, tag=corr_tag)
            writedlm(joinpath(outdir,
                        "G_k_$(L)_boot_DM_$(kl)_$(l)_epoch=$(epoch).dat"),
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
            outdir = correlation_dir(L, k; network=network, l=l, tag=corr_tag)
            savefig(p, joinpath(outdir, "propagator_3D_L$(L)_k$(k).pdf"))
            println("  ✓ Saved plot")
        end
    end

    return (k_vals=k_hmc, G_hmc=G_hmc, G_hmc_err=G_hmc_err,
            dm_results=dm_results)
end

# ═══════════════════════════════════════════════════════════════════
#  Parameter sets to process
#
#  Edit the list below and run the script.
#  Set epoch_list = nothing to auto-discover all available epochs,
#  or provide an explicit list like ["0009", "0049", "0099"].
# ═══════════════════════════════════════════════════════════════════

param_sets = [
    (L=64, k=0.2, network="ncsnpp", epoch_list=["0499", "0599", "4999", "19999"],
     data_subdir="data", corr_tag=""),
]

param_sets = [
    (L=64, k=0.1923, network="ncsnpp", epoch_list=["0049","0199","0999", "9999","19999"],
     data_subdir="data", corr_tag=""),
]

for ps in param_sets
    compute_propagators(;
        L           = ps.L,
        k           = ps.k,
        network     = ps.network,
        epoch_list  = ps.epoch_list,
        data_subdir = ps.data_subdir,
        corr_tag    = ps.corr_tag,
    )
end

V1=readdlm(joinpath(correlation_dir(64, 0.2; network="ncsnpp", l=0.9), "G_k_64_boot_HMC_2_0.9.dat"))
plot(V1[2:end,1], V1[2:end,2], yerr=V1[2:end,3], seriestype=:scatter, xaxis=:log, xlabel="p", ylabel="G(p)", title="3D  L=64, κ=0.2, λ=0.9", label="HMC")
