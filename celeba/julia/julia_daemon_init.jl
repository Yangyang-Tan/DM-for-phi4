# Persistent Julia server for propagator / analysis work.
#
# Start once:
#   nohup julia /data/tyywork/DM/celeba/julia_daemon_init.jl > /tmp/jdaemon.log 2>&1 &
#
# Send work via client (~2-3 s client start, vs ~30-60 s cold start):
#   julia -e 'using DaemonMode; runargs(3001)' /path/to/script.jl
#
# Packages pre-loaded below so subsequent client scripts start instantly.

using DaemonMode
using NPZ
using DelimitedFiles
using Plots
using Printf
using Random
using Statistics
using FFTW

# Shared propagator utilities (CPU + GPU versions already in the repo)
include("/data/tyywork/DM/2Dphi4/CorrelationUtils.jl")

# Uncomment to also pre-load the GPU path (~2-3s CUDA init):
# using CUDA
# include("/data/tyywork/DM/2Dphi4/CorrelationUtilsGPU.jl")

println("daemon ready: port 3001, pre-loaded: FFTW, NPZ, Plots, Statistics, ",
        "Random, DelimitedFiles, Printf, CorrelationUtils")
println("from a client:  julia -e 'using DaemonMode; runargs(3001)' script.jl")
flush(stdout)   # serve() blocks; without this the log stays empty forever

serve(3001)
