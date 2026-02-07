import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from hpc.nbody import NBodySimulator
import numpy as np
import os

# Force single thread for benchmark accuracy
os.environ['OMP_NUM_THREADS'] = '1'

print(f"Benchmarking Barnes-Hut JIT with N=5000...")
try:
    # Initialize
    t0 = time.time()
    # verify JIT compilation happens here
    sim = NBodySimulator(n_particles=5000, use_barnes_hut=True)
    init_time = time.time() - t0
    print(f"Initialization: {init_time:.2f}s")

    # Warmup (triggers JIT compilation)
    print("Warming up (compiling)...")
    t_warm = time.time()
    sim.step()
    print(f"Warmup step: {time.time() - t_warm:.2f}s")

    # Measure
    times = []
    print("Measuring...")
    for i in range(5):
        start = time.time()
        sim.step()
        dt = time.time() - start
        times.append(dt)
        print(f"Step {i+1}: {dt:.4f}s")

    avg_step = sum(times) / len(times)
    print(f"Average step time: {avg_step:.4f}s")
    
except Exception as e:
    import traceback
    traceback.print_exc()
