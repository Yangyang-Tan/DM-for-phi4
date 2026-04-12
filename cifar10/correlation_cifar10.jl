# Correlation analysis for CIFAR-10 grayscale images
# Computes momentum space propagator and radial averages
# Compatible with data saved from train_cifar10.py

using NPZ
using Statistics
using FFTW
using Plots

# ==================== Utility Functions ====================

function lattice_k_sq(nx, ny, L)
    """Compute lattice momentum squared for given indices."""
    kx = 2π * nx / L
    ky = 2π * ny / L
    # Lattice dispersion relation
    return 4 * sin(kx / 2)^2 + 4 * sin(ky / 2)^2
end

function diagonality_2d(nx, ny, L)
    """
    Compute diagonality parameter: sum(p_i^4) / (sum(p_i^2))^2

    In 2D:
    - Most diagonal (1,1) -> (1+1)/(2)^2 = 0.5
    - Most axial (1,0) -> 1/1 = 1

    Lower values = more diagonal (better for avoiding lattice artifacts)
    """
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

function get_momentum_propagator(cfgs::Array{T,3}) where T
    """
    Compute momentum space propagator G(k) = <|φ̃(k)|²>
    Input: cfgs with shape (Lx, Ly, N) where N is number of samples
    """
    Lx, Ly, N = size(cfgs)
    V = Lx * Ly

    G_k_sum = zeros(Lx, Ly)

    for i in 1:N
        cfg = @view cfgs[:, :, i]
        cfgc = cfg .- mean(cfg)  # Subtract mean (zero mode)
        # Fourier transform
        phi_k = fft(cfgc) ./ sqrt(V)  # Normalized
        # |φ̃(k)|²
        G_k = abs2.(phi_k)
        G_k_sum .+= G_k
    end

    G_k_mean = G_k_sum ./ N
    return G_k_mean
end

function radial_average_2d(G_k::Array{T,2}; max_diagonality::Float64=1.0) where {T}
    """
    Radially average G(k) to get G(|k|).

    Args:
        G_k: 2D momentum space propagator
        max_diagonality: Maximum diagonality parameter (0.5 = most diagonal, 1.0 = all points)
            - 0.5: only keep most diagonal momenta (1,1), (2,2), etc.
            - 0.6: keep slightly off-diagonal
            - 1.0: keep all momentum points (no filtering)

    Returns (k_values, G_averaged, counts)
    """
    Lx, Ly = size(G_k)
    L = Lx

    k_sq_list = Float64[]
    G_list = Float64[]

    n_total = 0
    n_filtered = 0

    for nx in 0:L-1, ny in 0:L-1
        mx = nx > L÷2 ? nx - L : nx
        my = ny > L÷2 ? ny - L : ny

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

    if max_diagonality < 1.0
        println("  Diagonality filter: kept $(n_total - n_filtered)/$n_total points (max_diag=$max_diagonality)")
    end

    return k_vals, G_avg, counts
end

# ==================== Load and Analyze ====================

# Configuration

class_name = "cat"  # Change this to match your trained class
L = 32

max_diag = 0.6

train_path = "data/cifar10_$(class_name)_train_$(L)x$(L).npy"
gen_path = "data/cifar10_$(class_name)_dm_$(L)x$(L)-7899.npy"
gen_path2 = "data/cifar10_$(class_name)_dm_$(L)x$(L)-199.npy"
gen_path3 = "data/cifar10_$(class_name)_dm_$(L)x$(L)-499.npy"
gen_path4 = "data/cifar10_$(class_name)_dm_$(L)x$(L)-49.npy"
gen_path5 = "data/cifar10_$(class_name)_dm_$(L)x$(L)-5199.npy"
# Load data
cfgs_train = npzread(train_path)
cfgs_dm = npzread(gen_path)
cfgs_dm2=npzread(gen_path2)
cfgs_dm3=npzread(gen_path3)
cfgs_dm4=npzread(gen_path4)
cfgs_dm5=npzread(gen_path5)
# Compute propagators
G_k_train = get_momentum_propagator(cfgs_train)
G_k_dm = get_momentum_propagator(cfgs_dm)
G_k_dm2 = get_momentum_propagator(cfgs_dm2)
G_k_dm3 = get_momentum_propagator(cfgs_dm3)
G_k_dm4 = get_momentum_propagator(cfgs_dm4)
G_k_dm5 = get_momentum_propagator(cfgs_dm5)
# Radial averages with diagonality filter
k_train, G_train, counts_train = radial_average_2d(G_k_train, max_diagonality=0.51)
k_dm, G_dm, counts_dm = radial_average_2d(G_k_dm, max_diagonality=0.51)
k_dm2, G_dm2, counts_dm2 = radial_average_2d(G_k_dm2, max_diagonality=0.51)
k_dm3, G_dm3, counts_dm3 = radial_average_2d(G_k_dm3, max_diagonality=0.51)
k_dm4, G_dm4, counts_dm4 = radial_average_2d(G_k_dm4, max_diagonality=0.51)
k_dm5, G_dm5, counts_dm5 = radial_average_2d(G_k_dm5, max_diagonality=0.51)
# Plot 1: G(k) comparison with diagonality filter
p1 = plot(
    k_train[2:end], G_train[2:end],
    seriestype=:scatter,
    label="Training Data",
    xlabel="|k|",
    ylabel="G(k)",
    title="CIFAR-10 $(class_name) - Propagator (Almost Axial)",
    xaxis=:log,
    legend=:topright,
    markersize=4
)
plot!(
    k_dm[2:end], G_dm[2:end],
    seriestype=:scatter,
    label="Generated (DM-7899)",
    markersize=4
)
plot!(
    k_dm2[2:end], G_dm2[2:end],
    seriestype=:scatter,
    label="Generated (DM-199)",
    markersize=4
)
plot!(
    k_dm3[2:end], G_dm3[2:end],
    seriestype=:scatter,
    label="Generated (DM-499)",
    markersize=4
)
plot!(
    k_dm4[2:end], G_dm4[2:end],
    seriestype=:scatter,
    label="Generated (DM-49)",
    markersize=4
)
plot!(
    k_dm5[2:end], G_dm5[2:end],
    seriestype=:scatter,
    label="Generated (DM-5199)",
    markersize=4
)


savefig(p1, "figures/cifar10_$(class_name)_propagator.png")

# Plot 2: G(k) comparison without filter (all points)
p1b = plot(
    k_train_all[2:end], G_train_all[2:end],
    seriestype=:scatter,
    label="Training Data",
    xlabel="|k|",
    ylabel="G(k)",
    title="CIFAR-10 $(class_name) - Propagator (no filter)",
    xaxis=:log,
    legend=:topright,
    markersize=3,
    alpha=0.6
)
plot!(
    k_dm_all[2:end], G_dm_all[2:end],
    seriestype=:scatter,
    label="Generated (DM)",
    markersize=3,
    alpha=0.6
)
savefig(p1b, "figures/cifar10_$(class_name)_propagator_all.png")

# Plot 3: Ratio G_dm / G_train
ratio = G_dm ./ G_train
p2 = plot(
    k_train[2:end], ratio[2:end],
    seriestype=:scatter,
    label="G_dm / G_train",
    xlabel="|k|",
    ylabel="Ratio",
    title="CIFAR-10 $(class_name) - Propagator Ratio (max_diag=$(max_diag))",
    xaxis=:log,
    legend=:topright,
    markersize=4
)
hline!([1.0], linestyle=:dash, color=:gray, label="Ratio = 1")
savefig(p2, "figures/cifar10_$(class_name)_propagator_ratio.png")

# Plot 4: Sample visualization (heatmaps)
p3 = heatmap(
    cfgs_train[:, :, 1],
    aspect_ratio=1.0,
    title="Training Sample",
    color=:grays
)
p4 = heatmap(
    cfgs_dm[:, :, 1],
    aspect_ratio=1.0,
    title="Generated Sample",
    color=:grays
)
p_samples = plot(p3, p4, layout=(1, 2), size=(800, 400))
savefig(p_samples, "figures/cifar10_$(class_name)_sample_comparison.png")

# Plot 5: FFT power spectrum comparison
fft_train = mean([fftshift(abs.(fft(cfgs_train[:, :, i]))) for i in 1:min(100, N_train)])
fft_dm = mean([fftshift(abs.(fft(cfgs_dm[:, :, i]))) for i in 1:min(100, N_dm)])

center = L ÷ 2 + 1
window = 10
p5 = heatmap(
    fft_train[center-window:center+window, center-window:center+window],
    aspect_ratio=1.0,
    title="FFT Power (Train)",
    color=:viridis
)
p6 = heatmap(
    fft_dm[center-window:center+window, center-window:center+window],
    aspect_ratio=1.0,
    title="FFT Power (DM)",
    color=:viridis
)
p_fft = plot(p5, p6, layout=(1, 2), size=(800, 400))
savefig(p_fft, "figures/cifar10_$(class_name)_fft_comparison.png")

# Plot 6: Diagonality visualization
# Show which k-points are kept/filtered
diag_map = zeros(L, L)
for nx in 0:L-1, ny in 0:L-1
    mx = nx > L÷2 ? nx - L : nx
    my = ny > L÷2 ? ny - L : ny
    diag_map[nx+1, ny+1] = diagonality_2d(mx, my, L)
end
p_diag = heatmap(
    fftshift(diag_map),
    aspect_ratio=1.0,
    title="Diagonality Map (threshold=$(max_diag))",
    color=:viridis,
    clim=(0, 1)
)
savefig(p_diag, "figures/cifar10_$(class_name)_diagonality_map.png")

println("\nPlots saved to figures/")
println("  - cifar10_$(class_name)_propagator.png (with diagonality filter)")
println("  - cifar10_$(class_name)_propagator_all.png (no filter)")
println("  - cifar10_$(class_name)_propagator_ratio.png")
println("  - cifar10_$(class_name)_sample_comparison.png")
println("  - cifar10_$(class_name)_fft_comparison.png")
println("  - cifar10_$(class_name)_diagonality_map.png")

# ==================== Summary Statistics ====================

println("\n" * "="^50)
println("Summary")
println("="^50)

# Compute mean ratio at different k ranges
low_k_idx = 2:min(5, length(ratio))
mid_k_idx = 6:min(15, length(ratio))
high_k_idx = 16:length(ratio)

if length(low_k_idx) > 0
    println("\nMean ratio (G_dm/G_train):")
    println("  Low |k|:  ", mean(ratio[low_k_idx]))
end
if length(mid_k_idx) > 0
    println("  Mid |k|:  ", mean(ratio[mid_k_idx]))
end
if length(high_k_idx) > 0
    println("  High |k|: ", mean(ratio[high_k_idx]))
end

println("\n" * "="^50)
println("Analysis Complete!")
println("="^50)
