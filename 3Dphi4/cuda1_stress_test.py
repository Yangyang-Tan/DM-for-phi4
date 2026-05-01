"""30-min real-workload stress test on cuda:1 using k=0.1923 trained model.
Runs SDE + ODE sampling in a loop. Writes NO files (sweep isn't on cuda:1).
Logs per-iteration time and GPU temp/power. Exits immediately on CUDA error.
"""
import sys, time, functools, subprocess, traceback
sys.path.append("..")
import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std

DEVICE = 'cuda:1'
CKPT = 'runs/phi4_3d_L32_k0.1923_l0.9_ncsnpp/models/epoch=10000.ckpt'
SIGMA = 750.0

def gpu_stat():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used',
         '--format=csv,noheader', '-i', '5'],
        capture_output=True, text=True).stdout.strip()
    return out

print(f"Loading model from {CKPT} ...", flush=True)
mp = functools.partial(marginal_prob_std, sigma=SIGMA)
net = NCSNpp3D(mp)
model = DiffusionModel3D.load_from_checkpoint(CKPT, score_model=net)
model = model.to(DEVICE).eval()
print(f"Model loaded on {DEVICE}  sigma={model.sigma}  L={model.L}", flush=True)

START = time.time()
DURATION = 30 * 60  # 30 minutes
iter_idx = 0
try:
    while time.time() - START < DURATION:
        iter_idx += 1
        t0 = time.time()
        # SDE
        t_sde = time.time()
        samples_em = model.sample(num_samples=512, num_steps=2000, schedule='log')
        torch.cuda.synchronize(DEVICE)
        sde_t = time.time() - t_sde
        stat_sde = gpu_stat()

        # ODE
        t_ode = time.time()
        samples_ode = model.sample_ode(num_samples=512, num_steps=400,
                                       schedule='log', method='dpm2')
        torch.cuda.synchronize(DEVICE)
        ode_t = time.time() - t_ode
        stat_ode = gpu_stat()

        elapsed_total = time.time() - START
        print(f"  [iter {iter_idx:3d}  t={elapsed_total/60:5.1f}m]  "
              f"SDE {sde_t:5.1f}s [{stat_sde}]  |  ODE {ode_t:5.1f}s [{stat_ode}]",
              flush=True)

        # cleanup
        del samples_em, samples_ode
        torch.cuda.empty_cache()

    print(f"\n✅ cuda:1 stress test PASSED: completed {iter_idx} iterations in {(time.time()-START)/60:.1f} min",
          flush=True)
    sys.exit(0)

except Exception as e:
    print(f"\n❌ cuda:1 stress test FAILED at iter {iter_idx}, t={(time.time()-START)/60:.1f}m:",
          flush=True)
    traceback.print_exc()
    print(f"gpu_stat at failure: {gpu_stat()}", flush=True)
    sys.exit(1)
