using LinearAlgebra, CUDA, Random, DifferentialEquations
using StatsBase,JLD2

cfgs = load("data/cfgs_k=0.5_l=2.5_32^3.jld2")["cfgs"]

cfgs = load("data/cfgs_k=0.14_l=0.022_32^3.jld2")["cfgs"]

cumulant
eachslice(cfgs, dims=[1,2])
testv=rand(2,2,3)
mapslices(x -> cumulant(x,2), cfgs, dims=3)

using Plots

cfgs=sol.u[12] |> Array
heatmap(mapslices(x -> cumulant(x, 4), cfgs, dims=4)[:,:,1,1], aspect_ratio=1.0)
