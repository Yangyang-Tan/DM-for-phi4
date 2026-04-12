plot(k_vals[2:end], 1 ./G_mean[2:end],
    seriestype=:scatter, xlabel="p", ylabel="1/G(p)",
    title="128^2, Broken phase", label="Data", markerstrokecolor=:auto)

plot(k_vals[2:end].^2, 1 ./ G_mean[2:end],
    seriestype=:scatter, xlabel="p^2", ylabel="1/G(p)",
    title="128^2, Broken phase", label="Data", markerstrokecolor=:auto)
using DelimitedFiles
datat10 = readdlm("data/G_k_128_jk_all.dat")
plot(datat10[2:end,1], 1 ./ datat10[2:end,2],
    seriestype=:scatter, xlabel="p", ylabel="1/G(p)",
    title="128^2, Broken phase", label="t=10", markerstrokecolor=:auto)

plot(datat10[2:end,1].^2, 1 ./ datat10[2:end,2],
    seriestype=:scatter, xlabel="p^2", ylabel="1/G(p)",
    title="128^2, Broken phase", label="t=10", markerstrokecolor=:auto)

writedlm("data/G_k_128_jk_HMC_t=inf.dat", [k_vals G_mean G_err])
