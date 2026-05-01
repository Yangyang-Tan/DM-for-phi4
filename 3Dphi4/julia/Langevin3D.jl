using LinearAlgebra, CUDA, Random, DifferentialEquations
using StatsBase

CUDA.device!(2)

myT=Float32
limitbound(a, n) =
    if a == n + 1
        1
    elseif a == 0
        n
    else
        a
    end


function update_langevin_3d!(
    dσ::AbstractArray{T,4},
    σ::AbstractArray{T,4},
    fun,
    κ::T,
    λ::T,
) where {T}
    N = size(σ, 1)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(σ)
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        x, y,z, k = Tuple(cind[i])
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
        @inbounds dσ[x, y, z, k] =
            2 * κ * (σ[xp1, y, z, k] + σ[xm1, y, z, k] + σ[x, yp1, z, k] + σ[x, ym1, z, k] + σ[x, y, zp1, k] + σ[x, y, zm1, k]) - fun(σ[x, y, z, k], κ, λ)
    end
    return nothing
end



function langevin_3d_loop_GPU(dσ, σ, fun, κ, λ)
    threads = 512
    blocks = 2^8
    @cuda blocks = blocks threads = threads maxregs = 4 update_langevin_3d!(dσ, σ, fun, κ, λ)
end


function modelA_3d_ODE_prob(;
    u0=error("u0 not provided"),
    tspan=myT.((0.0, 15.0)),
    T=1.0f0,
    # para = para::TaylorParameters,
    # dt = 0.1f0,
    # noise="coth",
    κ=0.5,
    λ=0.022,
    solver=DRI1NM(),
    save_start=false,
    save_everystep=false,
    save_end=false,
    abstol=5e-2,
    reltol=5e-2,
    args...,
)

    function Ufun(x, κ, λ)
        2 * (1 - 2 * λ) * x + 4 * λ * x^3
    end
    function g(du, u, p, t)
        du .= sqrt(2*T)
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_3d_loop_GPU(dσ, σ, Ufun, p[1], p[2])
        # CUDA.@sync @. dσ = (1 / γ) * dσ
    end
    sdeprob = SDEProblem(ODEfun_tex, g, u0_GPU, tspan, [κ, λ])
    # sdeprob = ODEProblem(ODEfun_tex, u0_GPU, tspan, [κ, λ])
    GC.gc(true)
    CUDA.reclaim()
    solve(
        sdeprob,
        # PCEuler(ggprime),
        solver,
        #DRI1(),
        #SRA3(),
        # SRIW1(),
        #RDI1WM(),
        #EM(),
        #dt = dt,
        save_start=save_start,
        save_everystep=save_everystep,
        save_end=save_end,
        abstol=abstol,
        reltol=reltol,
        # saveat=0.0:5.0:150.0,
        # callback = cb,
        # dt=0.01f0,
        args...,
    )
    # [saved_values.t saved_values.saveval]
end

# cb = SavingCallback(
#     (u, t, integrator) -> reshape(
#         mapslices(x -> cumulant.(Ref(abs.(x)), [1, 2, 3, 4]), Array(u); dims=[1, 2]),
#         (4, LangevinDynamics.M),
#     ),
#     saved_values;
#     # saveat=0.0:2.0:1500.0,
# )
saved_values = SavedValues(Float32, Any)
cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        # ϕ=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
        # return mean(ϕ)
        ϕ = mean(u, dims=[1, 2,3])[1, 1, 1, :]
        return mean(abs.(ϕ))
    end,
    saved_values;
    saveat=0.0:5.0:150.0,
    save_everystep=false,
    save_start=true
)

u0ini = randn(myT, 64, 64,64,512)
# u0ini = CUDA.fill(0.4f0, 32, 32, 32, 1)
CUDA.reclaim()
GC.gc(true)
sol = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.50f0, λ=2.5f0, T=1.0f0, save_end=true)
sol=Array(sol)

u0ini = randn(myT, 64, 64,64,512)
sol2 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol2=Array(sol2)

u0ini = randn(myT, 64, 64, 64, 512)
sol3 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol3 = Array(sol3)

u0ini = randn(myT, 64, 64, 64, 512)
sol4 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol4 = Array(sol4)

u0ini = randn(myT, 64, 64, 64, 512)
sol5 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol5 = Array(sol5)

u0ini = randn(myT, 64, 64, 64, 512)
sol6 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol6 = Array(sol6)

u0ini = randn(myT, 64, 64, 64, 512)
sol7 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol7 = Array(sol7)

u0ini = randn(myT, 64, 64, 64, 512)
sol8 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol8 = Array(sol8)

u0ini = randn(myT, 64, 64, 64, 512)
sol9 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol9 = Array(sol9)

u0ini = randn(myT, 64, 64, 64, 512)
sol10 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol10 = Array(sol10)

u0ini = randn(myT, 64, 64, 64, 512)
sol11 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol11 = Array(sol11)

u0ini = randn(myT, 64, 64, 64, 512)
sol12 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol12 = Array(sol12)

u0ini = randn(myT, 64, 64, 64, 512)
sol13 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol13 = Array(sol13)

u0ini = randn(myT, 64, 64, 64, 512)
sol14 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol14 = Array(sol14)

u0ini = randn(myT, 64, 64, 64, 512)
sol15 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol15 = Array(sol15)

u0ini = randn(myT, 64, 64, 64, 512)
sol16 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol16 = Array(sol16)

u0ini = randn(myT, 64, 64, 64, 512)
sol17 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol17 = Array(sol17)

u0ini = randn(myT, 64, 64, 64, 512)
sol18 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol18 = Array(sol18)

u0ini = randn(myT, 64, 64, 64, 512)
sol19 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol19 = Array(sol19)

u0ini = randn(myT, 64, 64, 64, 512)
sol20 = modelA_3d_ODE_prob(u0=u0ini, tspan=(0.0f0, 10.0f0), κ=0.5f0, λ=2.5f0, T=1.0f0, save_end=true)
sol20 = Array(sol20)

using Plots
# plot(saved_values.t, saved_values.saveval,label="κ=0.192, λ=0.9",title="<|φ|>,32^3")
# plot!(saved_values.t, saved_values.saveval,label="κ=0.182, λ=0.9",title="<|φ|>,32^3")
# plot!(saved_values.t, saved_values.saveval,label="κ=0.202, λ=0.9",title="<|φ|>,32^3")


heatmap(Array(sol[end][:,:,16, 17]),aspect_ratio=1.0)
plot(Array(mean(Array(stack(sol.u)), dims=[1, 2, 3])[1, 1, 1, :]))

# plot(mean(abs.(Array(stack(sol.u))), dims=[1, 2, 3])[1, 1, 1, 1:end])


plot!(mean(abs.(mean(Array(stack(sol.u)), dims=[1, 2])), dims=[3])[1, 1, 1, 1:end])
cfgs=reshape(stack(sol.u)[:,:,:,end],32,32,4096*16)|>Array

using JLD2
cfgs = cat(Array(sol), Array(sol2), Array(sol3), Array(sol4), Array(sol5), Array(sol6), Array(sol7), Array(sol8), Array(sol9), Array(sol10), Array(sol11), Array(sol12), Array(sol13), Array(sol14), Array(sol15), Array(sol16), Array(sol17), Array(sol18), Array(sol19), Array(sol20), dims=4)[:,:,:,:,end]

jldsave("DMasSQ-main/data/cfgs_k=0.21_l=0.022.jld2"; cfgs)

jldsave("data/cfgs_k=0.5_l=0.022.jld2"; cfgs)

jldsave("data/cfgs_k=0.21_l=0.022_small.jld2"; cfgs)

jldsave("data/cfgs_k=0.5_l=2.5_32^3_t=10.jld2"; cfgs)
jldsave("data/cfgs_k=0.5_l=2.5_64^3_t=10.jld2"; cfgs)


let usol = Array(stack(sol.u))
    plot(mean(var(usol,dims=[1,2]),dims=[1,2,3])[1,1,1,:])
end
var

abs.(stack(sol.u))
