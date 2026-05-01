# Propagator-mismatch diagnostics for CelebA 64×64 diffusion model.
#
# For each selected training checkpoint, load the pre-sampled images,
# compute the radially-averaged momentum propagator with bootstrap errors,
# and evaluate three complementary mismatch metrics against the training
# reference:
#   (1) per-mode Gaussian KL   D_k  = ½ (r - 1 - log r),  r = G_train/G_dm
#   (2) statistical z-score     z_k = (G_dm - G_train) / √(σ_dm² + σ_train²)
#   (3) phase-space weighted Δ  w_k = k^(D-1) (G_dm - G_train)  with D = 2
#
# Results and plots land in celeba_64_ncsnpp/correlation/.

ENV["GKSwstype"] = "100"   # headless GKS — write files, don't open display

using NPZ
using Statistics
using FFTW
using Plots
using DelimitedFiles
using Printf
gr()

include("../2Dphi4/CorrelationUtils.jl")

const BASE_DIR = joinpath(@__DIR__, "..", "runs")
const MAX_DIAG  = 0.51
const DIR_SYM   = :radial        # or :x, :y, :diagonal
const DIM       = 2              # images are 2D

# ═══════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════

model_dir(L, network) = joinpath(BASE_DIR, "celeba_$(L)_$(network)")
train_path(L, network) = joinpath(model_dir(L, network), "data", "celeba_train_$(L)x$(L).npy")

function dm_path(L, network, epoch; prefix::String="em")
    joinpath(model_dir(L, network), "data", "samples_$(prefix)_epoch=$(epoch).npy")
end

function diag_dir(L, network)
    d = joinpath(model_dir(L, network), "correlation")
    mkpath(d)
    return d
end

dir_suffix(direction) = direction == :radial ? "" : "_$(string(direction))"

# ═══════════════════════════════════════════════════════════════════
#  Propagator cache (computed once)
# ═══════════════════════════════════════════════════════════════════

function cached_propagator(npy_path, cache_path;
                           n_boot::Int=1000, direction::Symbol=:radial,
                           max_diagonality::Float64=MAX_DIAG)
    if isfile(cache_path)
        data = readdlm(cache_path)
        return data[:,1], data[:,2], data[:,3]
    end
    isfile(npy_path) || error("missing data: $npy_path")
    cfgs = npzread(npy_path)
    k, G, σ = propagator_radial_bootstrap(cfgs;
                    n_boot=n_boot, max_diagonality=max_diagonality,
                    direction=direction)
    writedlm(cache_path, [k G σ])
    return k, G, σ
end

# ═══════════════════════════════════════════════════════════════════
#  Main driver
# ═══════════════════════════════════════════════════════════════════

function run_diagnostics(; L::Int=64,
                           network::String="ncsnpp",
                           prefix::String="em",
                           direction::Symbol=DIR_SYM,
                           epoch_list::Vector{String},
                           n_boot::Int=1000,
                           k_floor::Float64=1e-4,
                           save::Bool=true,
    )

    outdir = diag_dir(L, network)
    ds     = dir_suffix(direction)
    println("\n", "="^70)
    println("  CelebA $(L)×$(L)  network=$network  sampler=$prefix  direction=$direction")
    println("="^70)

    # Training reference (cached)
    tcache = joinpath(outdir, "G_k_train_celeba_$(L)$(ds).dat")
    println("  → training propagator ($(isfile(tcache) ? "cached" : "fresh"))")
    k_t, G_t, σ_t = cached_propagator(train_path(L, network), tcache;
                                       n_boot=n_boot, direction=direction)

    # Keep only modes with |k| above k_floor (drop zero mode)
    keep = k_t .> k_floor
    k_t, G_t, σ_t = k_t[keep], G_t[keep], σ_t[keep]
    println("  → $(length(k_t)) non-zero momentum bins")

    summary_rows = Any[]
    D_k_per_epoch = Vector{Vector{Float64}}()   # [epoch_idx][k_bin]
    z_k_per_epoch = Vector{Vector{Float64}}()   # [epoch_idx][k_bin]

    # Plot handles (one per diagnostic, k-domain scatter)
    pkl = plot(xaxis=:log, yaxis=:log,
               xlabel="|k|", ylabel=raw"per-mode KL  D_k",
               title="CelebA $(L)×$(L) — per-mode KL",
               legend=:topright)
    pz  = plot(xaxis=:log,
               xlabel="|k|", ylabel=raw"z-score  (G_{DM}-G_{train}) / σ",
               title="CelebA $(L)×$(L) — statistical z-score",
               legend=:topright)
    pw  = plot(xaxis=:log,
               xlabel="|k|", ylabel=raw"k^{D-1} (G_{DM} - G_{train})",
               title="CelebA $(L)×$(L) — phase-space weighted Δ",
               legend=:topright)

    for epoch in epoch_list
        npy = dm_path(L, network, epoch; prefix=prefix)
        isfile(npy) || (@warn "missing sample file" path=npy; continue)

        dcache = joinpath(outdir,
            "G_k_DM_celeba_$(L)$(ds)_$(prefix)_epoch=$(epoch).dat")
        print("  → DM epoch=$epoch  ")
        k_d, G_d, σ_d = cached_propagator(npy, dcache;
                                           n_boot=n_boot, direction=direction)
        k_d, G_d, σ_d = k_d[keep], G_d[keep], σ_d[keep]
        println("done")

        # Sanity: shared k-grid
        @assert length(k_d) == length(k_t) && all(isapprox.(k_d, k_t))

        # (1) per-mode KL
        D_k, D_err, D_floor = per_mode_kl(G_d, G_t;
                                          G_dm_err=σ_d, G_train_err=σ_t)

        # (2) z-score
        z_k = propagator_zscore(G_d, G_t, σ_d, σ_t)

        # (3) phase-space weighted Δ
        w_k, w_err = phase_space_weighted_delta(k_t, G_d, G_t; D=DIM,
                                                G_dm_err=σ_d, G_train_err=σ_t)

        # Save per-epoch tabulated diagnostics
        if save
            fname = joinpath(outdir,
                "diag_celeba_$(L)$(ds)_$(prefix)_epoch=$(epoch).dat")
            writedlm(fname, [k_t G_d σ_d D_k D_err D_floor z_k w_k w_err],
                     ';')
        end

        # Aggregate scalars — total KL and KL split into |k|-tertiles
        kc_low   = quantile(k_t, 1/3)
        kc_high  = quantile(k_t, 2/3)
        ir_mask  = k_t .<  kc_low
        mid_mask = (k_t .>= kc_low) .& (k_t .< kc_high)
        uv_mask  = k_t .>= kc_high
        KL_tot   = sum(D_k)
        KL_ir    = sum(D_k[ir_mask])
        KL_mid   = sum(D_k[mid_mask])
        KL_uv    = sum(D_k[uv_mask])
        z_rms    = sqrt(mean(z_k.^2))
        w_L1     = sum(abs.(w_k))

        push!(summary_rows,
              (epoch=epoch, KL_total=KL_tot, KL_IR=KL_ir, KL_mid=KL_mid,
               KL_UV=KL_uv, z_rms=z_rms, dPhi2_L1=w_L1))
        push!(D_k_per_epoch, D_k)
        push!(z_k_per_epoch, z_k)

        # Overlay on plots
        lbl = "ep=$(epoch)"
        # KL: replace non-positive with NaN for log-y plot
        D_plot = [d > 0 ? d : NaN for d in D_k]
        plot!(pkl, k_t, D_plot; yerr=D_err, label=lbl,
              seriestype=:scatter, markerstrokecolor=:auto, markersize=3)
        plot!(pz,  k_t, z_k; label=lbl,
              seriestype=:scatter, markerstrokecolor=:auto, markersize=3)
        plot!(pw,  k_t, w_k; yerr=w_err, label=lbl,
              seriestype=:scatter, markerstrokecolor=:auto, markersize=3)
    end

    # Reference lines
    hline!(pz, [-2, 2]; linestyle=:dash, color=:gray, label="±2σ")
    hline!(pz, [0];    linestyle=:dot,  color=:black, label="")
    hline!(pw, [0];    linestyle=:dot,  color=:black, label="")

    if save
        savefig(pkl, joinpath(outdir, "diag_per_mode_kl_celeba_$(L)$(ds)_$(prefix).pdf"))
        savefig(pz,  joinpath(outdir, "diag_zscore_celeba_$(L)$(ds)_$(prefix).pdf"))
        savefig(pw,  joinpath(outdir, "diag_psweighted_celeba_$(L)$(ds)_$(prefix).pdf"))
        println("  ✓ plots saved to $outdir")
    end

    # Combined 3-panel plot
    pcomb = plot(pkl, pz, pw; layout=(1,3), size=(1600, 450),
                 left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    if save
        savefig(pcomb, joinpath(outdir,
            "diag_3panel_celeba_$(L)$(ds)_$(prefix).pdf"))
    end

    # Summary table
    if !isempty(summary_rows)
        println("\n  Summary (smaller = better)")
        println("  " * "─"^78)
        @printf "  %-8s %-11s %-11s %-11s %-11s %-9s %-12s\n" "epoch" "KL_total" "KL_IR" "KL_mid" "KL_UV" "z_rms" "|Δφ²|_L1"
        for r in summary_rows
            @printf "  %-8s %-11.4g %-11.4g %-11.4g %-11.4g %-9.3f %-12.4g\n" r.epoch r.KL_total r.KL_IR r.KL_mid r.KL_UV r.z_rms r.dPhi2_L1
        end

        if save
            fn = joinpath(outdir, "diag_summary_celeba_$(L)$(ds)_$(prefix).dat")
            open(fn, "w") do io
                println(io, "epoch  KL_total  KL_IR  KL_mid  KL_UV  z_rms  dPhi2_L1")
                for r in summary_rows
                    @printf io "%s  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g\n" r.epoch r.KL_total r.KL_IR r.KL_mid r.KL_UV r.z_rms r.dPhi2_L1
                end
            end
            println("  ✓ summary saved to $fn")
        end

        # Evolution plots: these are the main scientific deliverable
        if save
            plot_evolution(summary_rows, D_k_per_epoch, z_k_per_epoch, k_t;
                           outdir=outdir, L=L, ds=ds, prefix=prefix)
        end
    end

    return summary_rows
end

# ═══════════════════════════════════════════════════════════════════
#  Evolution plots: per-mode KL at selected k, aggregates vs epoch
# ═══════════════════════════════════════════════════════════════════

function plot_evolution(summary_rows, D_k_per_epoch, z_k_per_epoch, k_t;
                        outdir::String, L::Int, ds::String, prefix::String)

    # Parse epoch labels to integers; the "epoch=0000" checkpoint is the 1st
    # trained epoch, so we use +1 for the x-axis so log-scale works at epoch 1.
    eps = [parse(Int, r.epoch) + 1 for r in summary_rows]

    KL_tot = [r.KL_total for r in summary_rows]
    KL_ir  = [r.KL_IR    for r in summary_rows]
    KL_mid = [r.KL_mid   for r in summary_rows]
    KL_uv  = [r.KL_UV    for r in summary_rows]
    zrms   = [r.z_rms    for r in summary_rows]
    wL1    = [r.dPhi2_L1 for r in summary_rows]

    # Same k bins, geometrically spaced, used for both D_k and z_k evolution
    # so the two panels are directly comparable.
    nb  = length(k_t)
    sel = unique(clamp.(round.(Int, exp.(range(log(1), log(nb); length=5))), 1, nb))
    k_labels = [@sprintf("|k|=%.3f", k_t[ki]) for ki in sel]

    # (A) D_k vs epoch for selected k bins
    pA = plot(xaxis=:log, yaxis=:log,
              xlabel="epoch", ylabel=raw"D_k (per-mode KL)",
              title="CelebA $(L)×$(L) — D_k evolution at selected |k|",
              legend=:topright)
    for (i, ki) in enumerate(sel)
        vals = [D_k_per_epoch[j][ki] for j in 1:length(D_k_per_epoch)]
        vals_plot = [v > 0 ? v : NaN for v in vals]
        plot!(pA, eps, vals_plot;
              seriestype=:scatter, markerstrokecolor=:auto, markersize=4,
              label=k_labels[i])
        plot!(pA, eps, vals_plot; label="", linewidth=1, alpha=0.4)
    end

    # (B) KL aggregates vs epoch (total / UV / mid / IR)
    pB = plot(xaxis=:log, yaxis=:log,
              xlabel="epoch", ylabel=raw"\Sigma_k D_k",
              title="CelebA $(L)×$(L) — KL aggregates vs epoch",
              legend=:topright)
    for (yvals, lbl, mk) in [(KL_tot,"total",:circle), (KL_uv,"UV",:utriangle),
                             (KL_mid,"mid",:diamond), (KL_ir,"IR",:dtriangle)]
        plot!(pB, eps, yvals; seriestype=:scatter, markerstrokecolor=:auto,
              markersize=5, marker=mk, label=lbl)
        plot!(pB, eps, yvals; label="", linewidth=1, alpha=0.4)
    end

    # (C) z_k vs epoch for the same selected k bins — k-resolved z-score,
    # replaces the aggregate z_rms panel in the main grid.
    pC = plot(xaxis=:log,
              xlabel="epoch",
              ylabel=raw"z_k  (G_{DM}-G_{train}) / \sigma",
              title="CelebA $(L)×$(L) — z-score evolution at selected |k|",
              legend=:topright)
    for (i, ki) in enumerate(sel)
        vals = [z_k_per_epoch[j][ki] for j in 1:length(z_k_per_epoch)]
        plot!(pC, eps, vals;
              seriestype=:scatter, markerstrokecolor=:auto, markersize=4,
              label=k_labels[i])
        plot!(pC, eps, vals; label="", linewidth=1, alpha=0.4)
    end
    hline!(pC, [0.0]; linestyle=:dot,  color=:black, label="")
    hline!(pC, [-2.0, 2.0]; linestyle=:dash, color=:gray, label="")

    # (D) |Δ⟨φ²⟩|_L1 vs epoch
    pD = plot(xaxis=:log, yaxis=:log,
              xlabel="epoch", ylabel=raw"\Sigma_k k^{D-1}|G_{DM}-G_{train}|",
              title="CelebA $(L)×$(L) — |Δ⟨φ²⟩|_{L1} vs epoch",
              legend=false)
    plot!(pD, eps, wL1; seriestype=:scatter, markerstrokecolor=:auto,
          markersize=5)
    plot!(pD, eps, wL1; linewidth=1, alpha=0.4)

    # z_rms standalone — kept as reference but no longer in the main grid
    pZrms = plot(xaxis=:log,
                 xlabel="epoch", ylabel=raw"z_{rms}",
                 title="CelebA $(L)×$(L) — z-score RMS vs epoch",
                 legend=false)
    plot!(pZrms, eps, zrms; seriestype=:scatter, markerstrokecolor=:auto,
          markersize=5)
    plot!(pZrms, eps, zrms; linewidth=1, alpha=0.4)
    hline!(pZrms, [2.0]; linestyle=:dash, color=:gray, label="")
    hline!(pZrms, [1.0]; linestyle=:dot,  color=:gray, label="")

    savefig(pA,    joinpath(outdir, "evol_Dk_at_k_celeba_$(L)$(ds)_$(prefix).pdf"))
    savefig(pB,    joinpath(outdir, "evol_KL_bins_celeba_$(L)$(ds)_$(prefix).pdf"))
    savefig(pC,    joinpath(outdir, "evol_zk_at_k_celeba_$(L)$(ds)_$(prefix).pdf"))
    savefig(pD,    joinpath(outdir, "evol_dphi2_celeba_$(L)$(ds)_$(prefix).pdf"))
    savefig(pZrms, joinpath(outdir, "evol_zrms_celeba_$(L)$(ds)_$(prefix).pdf"))

    pgrid = plot(pA, pB, pC, pD; layout=(2,2), size=(1400, 950),
                 left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    savefig(pgrid, joinpath(outdir,
        "evol_4panel_celeba_$(L)$(ds)_$(prefix).pdf"))

    println("  ✓ evolution plots saved")
end

# ═══════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════

epoch_list = ["0001", "0002", "0004", "0006", "0011", "0017", "0028", "0045",
              "0095", "0151", "0200", "0351", "0559", "0890", "1417", "2257",
              "3944", "6280", "10000"]

run_diagnostics(
    L          = 64,
    network    = "ncsnpp",
    prefix     = "em",
    direction  = :radial,
    epoch_list = epoch_list,
    n_boot     = 500,
)
