# Compare generation quality at L=128 between old small-sigma and new
# large-sigma models, for k=0.2705 and k=0.28, at the last training epoch.
#
# Metrics:
#   1. per-sample variance ratio  ⟨φ²⟩_gen / ⟨φ²⟩_train
#   2. mean field offset          ⟨φ⟩_gen − ⟨φ⟩_train
#   3. propagator ratio           G_gen(k) / G_train(k)  (radial-averaged)
#   4. integrated Gaussian KL     Σ_k ½(r − 1 − log r),  r = G_train/G_gen
#
# Usage: julia compare_sigma_quality.jl

using NPZ
using HDF5
using DelimitedFiles
using Plots
using Printf
using Statistics

include("CorrelationUtils.jl")

const OUTDIR = joinpath(@__DIR__, "sigma_comparison_L128")
mkpath(OUTDIR)

function load_train(k)
    path = joinpath(@__DIR__,
        "trainingdata/cfgs_wolff_fahmc_k=$(k)_l=0.022_128^2.jld2")
    h5open(path, "r") do f
        cfgs = Float32.(read(f, "cfgs"))  # either (L,L,N) or (N,L,L)
        if size(cfgs, 1) != 128
            cfgs = permutedims(cfgs, (3, 2, 1))
        end
        return cfgs
    end
end

function load_gen(path)
    cfgs = npzread(path)  # (L, L, N)
    return Float32.(cfgs)
end

function moments(cfgs)
    # cfgs: (L, L, N)
    N = size(cfgs, ndims(cfgs))
    vals = vec(cfgs)
    return mean(vals), mean(vals.^2)
end

function compute_prop_cached(cfgs, cache_path; recompute=false)
    if isfile(cache_path) && !recompute
        println("  [cache] $cache_path")
        cached = readdlm(cache_path)
        return cached[:,1], cached[:,2], cached[:,3]
    end
    println("  computing propagator for N=$(size(cfgs,3)) configs ...")
    k_vals, G, G_err = propagator_radial_bootstrap(cfgs;
        n_boot=500, max_diagonality=0.51, direction=:radial, seed=42)
    writedlm(cache_path, [k_vals G G_err])
    return k_vals, G, G_err
end

function compare_one(k, old_sigma, new_sigma)
    println("\n" * "="^60)
    println("  k = $k    old σ = $old_sigma    new σ = $new_sigma")
    println("="^60)

    # Paths
    old_sample = joinpath(@__DIR__,
        "phi4_L128_k$(k)_l0.022_ncsnpp/data/samples_em_steps2000_epoch=9999.npy")
    new_sample = joinpath(@__DIR__,
        "phi4_L128_k$(k)_l0.022_ncsnpp_sigma$(new_sigma)/data/samples_em_steps2000_epoch=10000.npy")

    @assert isfile(old_sample) "missing: $old_sample"
    @assert isfile(new_sample) "missing: $new_sample"

    # Load all three
    println("loading training configs ...")
    train = load_train(k)
    println("  train shape: ", size(train))
    println("loading old-sigma samples ...")
    old = load_gen(old_sample)
    println("  old shape: ", size(old))
    println("loading new-sigma samples ...")
    new = load_gen(new_sample)
    println("  new shape: ", size(new))

    # Moments
    μ_t, m2_t = moments(train)
    μ_o, m2_o = moments(old)
    μ_n, m2_n = moments(new)
    σ2_t = m2_t - μ_t^2
    σ2_o = m2_o - μ_o^2
    σ2_n = m2_n - μ_n^2
    println("\n── moments (centered ⟨φ²⟩ = σ²) ──")
    @printf("  train  : ⟨φ⟩=%.4f  ⟨φ²⟩_c=%.4f\n", μ_t, σ2_t)
    @printf("  old σ=%d: ⟨φ⟩=%.4f  ⟨φ²⟩_c=%.4f   var_gen/var_train = %.4f\n",
            old_sigma, μ_o, σ2_o, σ2_o/σ2_t)
    @printf("  new σ=%d: ⟨φ⟩=%.4f  ⟨φ²⟩_c=%.4f   var_gen/var_train = %.4f\n",
            new_sigma, μ_n, σ2_n, σ2_n/σ2_t)

    # Propagators (cached)
    println("\n── propagators ──")
    kv, Gt, Gte = compute_prop_cached(train,
        joinpath(OUTDIR, "G_train_k$(k).dat"))
    _, Go, Goe = compute_prop_cached(old,
        joinpath(OUTDIR, "G_old_sigma$(old_sigma)_k$(k).dat"); recompute=true)
    _, Gn, Gne = compute_prop_cached(new,
        joinpath(OUTDIR, "G_new_sigma$(new_sigma)_k$(k).dat"); recompute=true)

    # Skip k=0 bin
    nz = kv .> 1e-8
    kvnz = kv[nz]
    Gt_nz, Gte_nz = Gt[nz], Gte[nz]
    Go_nz, Goe_nz = Go[nz], Goe[nz]
    Gn_nz, Gne_nz = Gn[nz], Gne[nz]
    ratio_old = Go_nz ./ Gt_nz
    ratio_new = Gn_nz ./ Gt_nz
    ratio_old_err = ratio_old .* sqrt.((Goe_nz ./ Go_nz).^2 .+ (Gte_nz ./ Gt_nz).^2)
    ratio_new_err = ratio_new .* sqrt.((Gne_nz ./ Gn_nz).^2 .+ (Gte_nz ./ Gt_nz).^2)

    # Integrated Gaussian KL
    D_old, _, _ = per_mode_kl(Go_nz, Gt_nz)
    D_new, _, _ = per_mode_kl(Gn_nz, Gt_nz)
    @printf("\n  max |ratio-1|:   old=%.4f   new=%.4f\n",
            maximum(abs.(ratio_old .- 1)), maximum(abs.(ratio_new .- 1)))
    @printf("  mean |ratio-1|:  old=%.4f   new=%.4f\n",
            mean(abs.(ratio_old .- 1)),   mean(abs.(ratio_new .- 1)))
    @printf("  Σ D_k (KL):      old=%.4f   new=%.4f\n",
            sum(D_old), sum(D_new))

    # Plot: G(k), ratio, KL
    p1 = plot(kvnz, Gt_nz; yerr=Gte_nz, seriestype=:scatter,
             xaxis=:log, yaxis=:log, label="train", color=:black,
             xlabel="|k|", ylabel="G(k)", markersize=3,
             title="CelebA-style G(k) comparison  k=$k")
    plot!(p1, kvnz, Go_nz; yerr=Goe_nz, seriestype=:scatter,
          label="old σ=$old_sigma", color=:orange, markersize=3)
    plot!(p1, kvnz, Gn_nz; yerr=Gne_nz, seriestype=:scatter,
          label="new σ=$new_sigma", color=:steelblue, markersize=3)

    p2 = plot(kvnz, ratio_old; yerr=ratio_old_err, seriestype=:scatter,
             xaxis=:log, label="old σ=$old_sigma", color=:orange,
             xlabel="|k|", ylabel="G_gen / G_train", markersize=3,
             title="Ratio (k=$k)")
    plot!(p2, kvnz, ratio_new; yerr=ratio_new_err, seriestype=:scatter,
          label="new σ=$new_sigma", color=:steelblue, markersize=3)
    hline!(p2, [1.0]; linestyle=:dash, color=:gray, label="")

    p3 = plot(kvnz, D_old; seriestype=:scatter, xaxis=:log, yaxis=:log,
             label="old σ=$old_sigma", color=:orange,
             xlabel="|k|", ylabel="per-mode KL D_k", markersize=3,
             title="Gaussian per-mode KL (k=$k)")
    plot!(p3, kvnz, D_new; seriestype=:scatter, label="new σ=$new_sigma",
          color=:steelblue, markersize=3)

    p4 = bar([1, 2, 3], [σ2_t, σ2_o, σ2_n];
             color=[:black, :orange, :steelblue],
             xticks=(1:3, ["train", "old σ=$old_sigma", "new σ=$new_sigma"]),
             ylabel="⟨φ²⟩_centered", title="Variance (k=$k)",
             legend=false)

    P = plot(p1, p2, p3, p4; layout=(2,2), size=(1200, 900))
    outfile = joinpath(OUTDIR, "sigma_compare_k$(k).pdf")
    savefig(P, outfile)
    savefig(P, replace(outfile, ".pdf" => ".png"))
    println("  ✓ saved $outfile")

    return Dict(
        :k => k, :old_sigma => old_sigma, :new_sigma => new_sigma,
        :var_ratio_old => σ2_o/σ2_t, :var_ratio_new => σ2_n/σ2_t,
        :kl_old => sum(D_old), :kl_new => sum(D_new),
        :mean_absratio_old => mean(abs.(ratio_old .- 1)),
        :mean_absratio_new => mean(abs.(ratio_new .- 1)),
    )
end

results = []
push!(results, compare_one(0.2705, 150, 450))
push!(results, compare_one(0.28,    60, 640))

println("\n\n" * "="^60)
println("  SUMMARY TABLE")
println("="^60)
@printf("%-10s %-12s %-12s %-16s %-16s %-14s %-14s\n",
        "k", "old σ", "new σ", "var_ratio_old", "var_ratio_new",
        "mean|Δ|_old", "mean|Δ|_new")
for r in results
    @printf("%-10.4f %-12d %-12d %-16.4f %-16.4f %-14.4f %-14.4f\n",
            r[:k], r[:old_sigma], r[:new_sigma],
            r[:var_ratio_old], r[:var_ratio_new],
            r[:mean_absratio_old], r[:mean_absratio_new])
end
