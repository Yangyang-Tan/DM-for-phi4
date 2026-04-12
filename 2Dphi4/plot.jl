using Plots
using NPZ


cfgs = npzread("2Dphi4/data/phi4_L128_k0.5_l0.022_t=0.npy")[:,1,1,:,:]
cfgs2 = npzread("2Dphi4/data/phi4_L128_k0.5_l0.022_t=0.5.npy")[:,1,1,:,:]
cfgs3 = npzread("2Dphi4/data/phi4_L128_k0.5_l0.022_t=1.npy")[:,1,1,:,:]
cfgs4 = npzread("2Dphi4/data/phi4_L128_k0.5_l0.022_t=0.05.npy")[:,1,1,:,:]
cfgs5 = npzread("2Dphi4/data/phi4_L128_k0.5_l0.022_t=0.2.npy")[:,1,1,:,:]

heatmap(cfgs[10,:,:],aspect_ratio=1.0)
heatmap(cfgs2[1000,:,:],aspect_ratio=1.0)
heatmap(cfgs3[10,:,:],aspect_ratio=1.0)
heatmap(cfgs4[10,:,:],aspect_ratio=1.0)
heatmap(cfgs5[5000,:,:],aspect_ratio=1.0)
