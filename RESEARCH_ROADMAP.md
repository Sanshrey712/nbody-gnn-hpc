# How to Scale Up on Limited Hardware (The "Hackers" Guide)

Scaling beyond **5,000 particles** on a single 12GB GPU is impossible with standard brute-force training. However, by trading **compute time** for **memory**, we can scale to **50,000+ particles**.

Here is the engineering roadmap to achieve this on your current architecture.

---

## 1. Gradient Checkpointing (The "Infinite Memory" Hack)

**Concept:** 
Standard backpropagation stores all intermediate activations ($O(L)$ layers).
Gradient checkpointing **throws away** intermediate activations during the forward pass and **recomputes** them during the backward pass.

**Gain:** Reduces memory from $O(L)$ to $O(\sqrt{L})$.
**Cost:** Increases training time by ~30-50%.

---

## 4. The "25k Particle" Run (Concrete Plan)

If you have **2TB Storage** and want to run a massive **25,000 Particle** simulation on your current hardware check this:

### A. The Configuration
*   **Particles:** 25,000
*   **Simulations:** 600
*   **Model:** GNN (Hidden: **128**, Layers: **3**, Neighbors: **20**)
    *   *Why smaller?* Gravity rules are simple; you don't need a huge brain, just a Huge Graph. 128 dim is plenty.

### B. The Memory Bottleneck (RAM)
*   **Problem:** 25k particles $\times$ 600 sims = ~300 GB Data. You have 32 GB RAM.
*   **Solution:** **Lazy Loading**. Do NOT load `h5py` files into RAM. Read from disk during training.
    ```python
    # In src/ai/train.py -> NBodyDataset.__getitem__
    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            # Read ONLY this specific sample from disk
            inputs = f['inputs'][idx]
            targets = f['targets'][idx]
        return torch.tensor(inputs), torch.tensor(targets)
    ```

### C. The Timeline (Total: ~3 Days)
1.  **Data Generation (CPU):** ~25 Hours.
    *   24 Workers on Ryzen 9.
2.  **Training (GPU):** ~50 Hours.
    *   100 Epochs @ 30 mins/epoch.
    *   Using Gradient Checkpointing + AMP.

**Verdict:** You can produce a **research-grade 25k particle simulation** in 3 days on your home hardware.

