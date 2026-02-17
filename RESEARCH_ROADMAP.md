# Research Roadmap: Incremental AI-HPC N-Body Simulation

This roadmap breaks down the advanced research goals into **incremental, manageable stages**. Each stage produces a complete, working result before moving to the next.

## Phase 1: The "Smart" Baseline (Weeks 1-2)
**Goal:** Fix the infinite-range gravity problem without expensive computation.
**Action:** Implement **Long-Range Force Approximation ("Mean Field")**.
*   **How:** Add a single "Global Node" to your existing graph. It connects to every particle and learns the average gravitational pull of the galaxy.
*   **Why start here?** It's the easiest change (adding 1 node type) and immediately stops distant stars from drifting.
*   **Result:** A model that works for 5000+ particles with high accuracy.

## Phase 2: The "Structure" Upgrade (Weeks 3-4)
**Goal:** Scale to 10,000+ particles efficiently.
**Action:** Implement **Hierarchical Graph Networks**.
*   **How:** Instead of just one Global Node (Phase 1), create a "Tree" of nodes. Cluster 100 distant stars into 1 Super-Node.
*   **Why second?** It builds on the logic of Phase 1 but makes it smarter. You use the same GNN principles but change *how* the graph is built.
*   **Result:** A professional-grade N-body code that scales linearly $O(N)$.

## Phase 3: The "SOTA" Physics (Weeks 5-6)
**Goal:** Perfect symmetry and massive data efficiency.
**Action:** Replace standard GNN layers with **E(n)-Equivariant Layers**.
*   **How:** Swap out your `MessagePassing` layers for `E(n)-GNN` layers (using `e3nn` or `egnn-pytorch`).
*   **Why third?** This changes the internal math of the network. It's better to do this *after* you have the right graph structure (Phase 2).
*   **Result:** State-of-the-Art accuracy. Your model will learn physics from 10x less data because it "knows" rotation symmetry perfectly.

---

## Phase 4: Discovery & Astrophysics (Long Term)

These are "Moonshots" to attempt once Phases 1-3 are solid.

### A. Neural ODEs
**Goal:** Continuous-time prediction ($t=3.14$).
**Action:** Wrap the Phase 3 model in an ODE Solver (`torchdiffeq`).

### B. Symbolic Regression
**Goal:** Rediscover Newton's Law.
**Action:** Run `PySR` on the trained weights of Phase 3 to extract the formula $F = G \frac{m_1 m_2}{r^2}$.

### C. Smoothed Particle Hydrodynamics (SPH)
**Goal:** Simulate Star Formation.
**Action:** Add "Gas" particles to the simulation. Requires writing a new ground-truth simulator.
