# QuotientFlows

**A Systematic Study of Normalizing Flows on Quotient Spaces for Lattice Field Theory**

This repository contains PyTorch implementations for studying how normalizing flows can learn distributions on quotient spaces arising from lattice field theories. The experiments systematically compare three sampling strategies across multiple physical systems: training flows on the full configuration space X, on quotients X/G by discrete symmetry groups G, and using gauge-fixing procedures for continuous symmetries.

---

## Table of Contents

1. [Overview](#overview)
2. [Physical Models](#physical-models)
3. [Repository Structure](#repository-structure)
4. [File Descriptions](#file-descriptions)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Experimental Workflow](#experimental-workflow)
8. [Theoretical Background](#theoretical-background)
9. [Results and Observables](#results-and-observables)
10. [Citation](#citation)

---

## Overview

Normalizing flows are generative models that learn to transform a simple base distribution (typically Gaussian) into a complex target distribution through a sequence of invertible transformations. In lattice field theory, configuration spaces often possess large symmetry groups (discrete or continuous), and one may ask whether learning on a reduced quotient space improves sampling efficiency.

This repository implements experiments to address this question across four physical systems:

| System | Symmetry Groups | Quotient Spaces Studied |
|--------|-----------------|-------------------------|
| Double-well (1D) | Z₂ (reflection) | X, X/Z₂ (equivariant), X/Z₂ (quotient) |
| φ⁴ theory (1D ring) | Z₂ × C_L (sign × translations) | X, X/Z₂, X/(Z₂×C_L) |
| φ⁴ theory (2D torus) | Z₂ × Λ (sign × translations) | X, X/Z₂, X/(Z₂×Λ) |
| U(1) gauge theory (2D) | Λ (translations), G_gauge (local gauge) | X, X/Λ, X/G_gauge |
| SU(2) gauge theory (2D) | Λ (translations) | X, X/Λ |

---

## Physical Models

### 1. Double-Well Potential (Experiment 0)

A one-dimensional scalar field with a double-well potential, modelled as a mixture of two Gaussians:

$$p(x) = \frac{1}{2}\mathcal{N}(x \mid -m, \sigma^2) + \frac{1}{2}\mathcal{N}(x \mid +m, \sigma^2)$$

This system has an exact Z₂ symmetry under x → −x.

### 2. φ⁴ Theory on a 1D Ring (Experiment 1)

Lattice scalar field theory with action:

$$S[\phi] = \sum_{i=0}^{L-1} \left[ \frac{1}{2}(\phi_{i+1} - \phi_i)^2 + \frac{m^2}{2}\phi_i^2 + \frac{\lambda}{4}\phi_i^4 \right]$$

with periodic boundary conditions (φ_{L} ≡ φ_0). Symmetries include:
- **Z₂**: Global sign flip φ → −φ
- **C_L**: Cyclic translations φ_i → φ_{i+k mod L}

### 3. φ⁴ Theory on a 2D Torus

Extension to two dimensions with action:

$$S[\phi] = \beta \sum_{\mathbf{x}} \left[ \frac{1}{2}\sum_{\mu=1,2}(\phi_{\mathbf{x}+\hat{\mu}} - \phi_{\mathbf{x}})^2 + \frac{m^2}{2}\phi_{\mathbf{x}}^2 + \frac{\lambda}{4}\phi_{\mathbf{x}}^4 \right]$$

Symmetries: Z₂ (global sign) × Λ (translation group on the L×L torus).

### 4. U(1) Gauge Theory (2D)

Two-dimensional compact U(1) lattice gauge theory with Wilson plaquette action:

$$S[\theta] = -\beta \sum_{\square} \cos(\theta_\square)$$

where θ_□ = θ_x(x,y) + θ_y(x+1,y) − θ_x(x,y+1) − θ_y(x,y) is the plaquette angle. Link variables θ_μ(x) ∈ (−π, π] represent U(1) group elements.

Symmetries studied:
- **Λ**: Lattice translations
- **G_gauge**: Local U(1) gauge transformations θ_μ(x) → θ_μ(x) + α(x) − α(x+μ̂)

### 5. SU(2) Gauge Theory (2D)

Two-dimensional SU(2) lattice gauge theory. Link variables are represented as unit quaternions (a_0, a_1, a_2, a_3) with ||a|| = 1. The Wilson action is:

$$S[U] = -\frac{\beta}{2}\sum_{\square} \mathrm{Tr}(U_\square) = -\beta\sum_{\square} a_0^{(\square)}$$

Symmetry studied: Λ (lattice translations).

---

## Repository Structure

```
QuotientFlows/
├── experiment0_double_well.py           # 1D double-well: vanilla, equivariant, quotient flows
├── experiment1_phi4_ring_hmc.py         # φ⁴ 1D ring: HMC sampling + canonicalisation
├── experiment1_phi4_ring_flows.py       # φ⁴ 1D ring: flow training on X, X/Z₂, X/(Z₂×C_L)
├── experiment_phi4_2d_hmc.py            # φ⁴ 2D: HMC sampling + canonicalisation
├── experiment_phi4_2d_flows.py          # φ⁴ 2D: flow training
├── experiment_u1_2d_hmc.py              # U(1) 2D: HMC + translation canonicalisation
├── experiment_u1_2d_flows.py            # U(1) 2D: flows on X and X/Λ
├── experiment_u1_2d_flows_b.py          # U(1) 2D: alternative implementation
├── experiment_u1_2d_hmc_b.py            # U(1) 2D: alternative HMC implementation
├── experiment_u1_2d_hmc_gaugequotient.py    # U(1) 2D: HMC + maximal-tree gauge fixing
├── experiment_u1_2d_flows_gaugequotient.py  # U(1) 2D: flows on X and X/G_gauge
├── experiment_u1_ring_1d_flows.py       # U(1) 1D ring: flow experiments
├── experiment_u1_ring_1d_hmc.py         # U(1) 1D ring: HMC sampling
├── experiment_su2_2d_hmc.py             # SU(2) 2D: Metropolis MC + canonicalisation
├── experiment_su2_2d_flows.py           # SU(2) 2D: flow training
├── experiment_su2_2d_flows_gaugequotient.py # SU(2) 2D: gauge quotient flows
├── experiment_su2_2d_hmc_gaugequotient.py   # SU(2) 2D: gauge quotient HMC
├── plot_phi4_ring_1d.py                 # Plotting: φ⁴ 1D results
├── plot_phi4_2d.py                      # Plotting: φ⁴ 2D results
├── plot_u1_2d.py                        # Plotting: U(1) 2D results
├── plot_u1_2d_b.py                      # Plotting: U(1) 2D alternative
├── plot_u1_2d_gaugequotient.py          # Plotting: U(1) gauge quotient results
├── plot_u1_ring_1d.py                   # Plotting: U(1) 1D ring results
├── plot_su2_2d.py                       # Plotting: SU(2) 2D results
└── plot_su2_2d_gaugequotient.py         # Plotting: SU(2) gauge quotient results
```

---

## File Descriptions

### Experiment 0: Double-Well (Pedagogical)

#### `experiment0_double_well.py`

A self-contained demonstration comparing three flow architectures on a Z₂-symmetric 1D target:

**Components:**
- `log_p_double_well(x)`: Target log-density for the Gaussian mixture
- `sample_double_well(n)`: Exact sampling from the target
- `ResidualTanhLayer1D`: Unconstrained residual layer y = x + a·tanh(bx + c)
- `ResidualTanhOddLayer1D`: Z₂-equivariant layer y = x + a·tanh(bx) ensuring y(−x) = −y(x)
- `VanillaFlow1D`: Composition of unconstrained layers
- `EquivariantFlow1D`: Composition of odd layers (Z₂-equivariant map)
- `QuotientFlow1D`: Flow on |x| ≥ 0 with random sign lifting

**Training:** Reverse KL minimisation with Wasserstein-1 distance tracking.

**Outputs:**
- `experiment0_double_well_histograms.png`: Sample histograms
- `experiment0_double_well_training_curves.png`: W₁ distance vs epoch

---

### Experiment 1: φ⁴ Theory on 1D Ring

#### `experiment1_phi4_ring_hmc.py`

Hybrid Monte Carlo sampler for the 1D φ⁴ theory.

**Class `Phi4Ring1D`:**
- `action(phi)`: Computes S[φ] for configuration φ ∈ ℝ^L
- `grad_action(phi)`: Analytical gradient dS/dφ

**Canonicalisation Functions:**
- `canonicalise_Z2(phi)`: If ⟨φ⟩ < 0, flip φ → −φ
- `canonicalise_Z2_translations(phi)`: Find i* = argmax|φ_i|, flip if φ_{i*} < 0, then roll to place i* at index 0

**Outputs:** `phi4_ring_1d_L{L}_m2{m2}_lam{lam}.npz` containing raw, Z₂-canonical, and fully-canonical configurations.

#### `experiment1_phi4_ring_flows.py`

Trains RealNVP flows on configurations from the HMC ensemble.

**Architecture:**
- `MLPConditioner`: 2-layer MLP outputting scale and shift parameters
- `RealNVPCoupling`: Mask-based affine coupling layer
- `RealNVPFlow1D`: Composition with alternating binary masks

**Experimental Design:**
1. Train flow on X (raw configurations)
2. Train flow on X/Z₂ (Z₂-canonicalised)
3. Train flow on X/(Z₂×C_L) (fully canonicalised)

**Lifting Procedure:** For quotient flows, samples are lifted back to X by:
- X/Z₂: Apply random global sign ±1
- X/(Z₂×C_L): Apply random sign and random cyclic shift

**Observables:** ⟨φ²⟩, ⟨φ⁴⟩, ⟨M⟩, ⟨|M|⟩, Binder cumulant U₄

---

### φ⁴ Theory on 2D Torus

#### `experiment_phi4_2d_hmc.py`

HMC for the 2D φ⁴ theory on an L×L periodic lattice.

**Class `Phi4Lattice2D`:**
- Kinetic term: (1/2)Σ_μ(φ_{x+μ̂} − φ_x)²
- Action and gradient computed with periodic boundary conditions

**Canonicalisation:**
- `canonicalise_Z2(phi)`: Global sign based on mean
- `canonicalise_Z2_translations(phi)`: Uses argmax(|φ|) for translation fixing

#### `experiment_phi4_2d_flows.py`

Identical architecture to the 1D case, scaled to dimension D = L×L.

---

### U(1) Gauge Theory (2D)

#### `experiment_u1_2d_hmc.py`

HMC sampler for 2D U(1) lattice gauge theory.

**Class `U1Gauge2D`:**
- `plaquette_angles(theta)`: Computes θ_□ = θ_x + θ_y^{+x} − θ_x^{+y} − θ_y
- `action(theta)`: Wilson action S = −β Σ cos(θ_□)
- `grad_action(theta)`: Analytical staple-based gradient

**Canonicalisation (Translation Quotient):**
`canonicalise_translations(theta)`: Uses weighted centre-of-mass of plaquette energy density E(x) = 1 − cos(θ_□(x)) to determine translation shift. More continuous than argmax.

**Observables:**
- Plaquette energy: ⟨E_□⟩ = ⟨1 − cos θ_□⟩
- Polyakov loop: |P| where P = L⁻¹ Σ_j exp(i Σ_i θ_x(i,j))

#### `experiment_u1_2d_flows.py`

RealNVP flows on flattened link configurations θ ∈ ℝ^{2L²}.

**Workflow:**
1. Load HMC data (raw and translation-canonical)
2. Train flows on X and X/Λ
3. Sample, wrap angles to (−π, π], apply random translations for lifting
4. Compute observables

#### `experiment_u1_2d_hmc_gaugequotient.py`

HMC with **maximal-tree gauge fixing** for the local U(1) gauge quotient.

**Gauge Fixing Algorithm:**
```
gauge_fix_maximal_tree(theta):
    1. Build spanning tree via BFS from site (0,0)
    2. For each tree edge, compute site phase α(x) such that θ'_link = 0
    3. Apply gauge transformation to ALL links (including periodic wraps)
    4. Remaining links carry gauge-invariant plaquette flux
```

This reduces the configuration space from 2L² link angles to L² independent degrees of freedom (one per plaquette).

#### `experiment_u1_2d_flows_gaugequotient.py`

Trains flows on gauge-fixed configurations with appropriate lifting (random local gauge transformations).

---

### SU(2) Gauge Theory (2D)

#### `experiment_su2_2d_hmc.py`

Metropolis Monte Carlo for 2D SU(2) gauge theory. (HMC is more complex for non-Abelian groups; Metropolis suffices for exploratory studies.)

**Quaternion Representation:**
SU(2) elements are represented as unit quaternions q = (a_0, a_1, a_2, a_3) with ||q|| = 1.

**Key Functions:**
- `su2_quat_mul(a, b)`: Quaternion multiplication
- `su2_quat_inv(a)`: Quaternion conjugate (inverse for unit quaternions)
- `su2_quat_exp(eps, n)`: Small group element exp(iε n·σ)
- `su2_plaquette_field(U)`: Computes a_0^{(□)} for each plaquette

**Canonicalisation:** Translation fixing via argmax of plaquette scalar part.

#### `experiment_su2_2d_flows.py`

RealNVP flows on flattened quaternion representations (D = 2×L×L×4).

**Lifting to SU(2):**
After sampling from the flow, each 4-vector block is normalised to unit length:
```python
U = su2_quat_normalize(x.view(N, 2, L, L, 4))
```

---

### Plotting Scripts

All plotting scripts follow a consistent pattern:

1. Load HMC reference data and flow results from `.npz` files
2. Generate figures:
   - Training NLL curves (flow on X vs quotients)
   - NLL gap: ΔNLL = NLL_X − NLL_{X/G}
   - Observable histograms (HMC vs flow samples)
   - Histogram differences

**Output:** PNG figures saved to `figures_{model}_{params}/` directories.

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (with CUDA/MPS support recommended)
- NumPy
- SciPy (for Wasserstein distance in Experiment 0)
- Matplotlib
- tqdm

### Setup

```bash
# Clone repository
git clone https://github.com/QCD-Internship/QuotientFlows.git
cd QuotientFlows

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install torch numpy scipy matplotlib tqdm
```

### Device Support

All scripts automatically detect and use:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Metal on M1/M2/M3 Macs)
3. CPU (fallback)

---

## Usage

### Running a Complete Experiment

Each physical system requires two steps: (1) generate reference data via HMC/MC, (2) train flows.

**Example: 1D φ⁴ Ring**

```bash
# Step 1: Generate HMC samples
python experiment1_phi4_ring_hmc.py
# Creates: phi4_ring_1d_L32_m2-0.50_lam3.00.npz

# Step 2: Train flows
python experiment1_phi4_ring_flows.py
# Creates: phi4_ring_1d_L32_m2-0.50_lam3.00_flows_results.npz
#          phi4_ring_1d_L32_m2-0.50_lam3.00_training_nll.png

# Step 3: Generate publication plots
python plot_phi4_ring_1d.py
# Creates: figures_phi4_ring_1d_L32_m2-0.50_lam3.00/*.png
```

**Example: 2D U(1) with Gauge Quotient**

```bash
# Generate gauge-fixed HMC samples
python experiment_u1_2d_hmc_gaugequotient.py --L 16 --beta 2.0

# Train flows on X and X/G_gauge
python experiment_u1_2d_flows_gaugequotient.py --file u1_2d_L16_beta2.00_gaugequotient.npz

# Plot results
python plot_u1_2d_gaugequotient.py
```

### Command-Line Arguments

The gauge quotient scripts accept command-line arguments:

```bash
python experiment_u1_2d_hmc_gaugequotient.py \
    --L 16 \            # Lattice size
    --beta 2.0 \        # Gauge coupling
    --n-therm 2000 \    # Thermalisation steps
    --n-samples 2000 \  # Production samples
    --n-skip 5 \        # Thinning interval
    --eps 0.1 \         # Leapfrog step size
    --n-leapfrog 10 \   # Leapfrog steps per trajectory
    --seed 123          # Random seed
```

---

## Experimental Workflow

### Standard Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  HMC/Metropolis │────▶│   Canonicalisation   │────▶│   Save .npz     │
│    Sampling     │     │  (quotient mapping)  │     │   (raw, canon)  │
└─────────────────┘     └──────────────────────┘     └────────┬────────┘
                                                              │
                        ┌─────────────────────────────────────┘
                        ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Train Flows    │────▶│  Sample & Lift to X  │────▶│   Compute Obs   │
│  (X and X/G)    │     │  (apply random g∈G)  │     │   (compare)     │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
```

### Canonicalisation Procedures

| Symmetry | Canonicalisation Rule | Lifting Procedure |
|----------|----------------------|-------------------|
| Z₂ (sign) | Flip if ⟨φ⟩ < 0 | Random sign ±1 |
| C_L (translations) | Roll to place argmax at origin | Random cyclic shift |
| Λ (2D translations) | Centre-of-mass shift | Random 2D translation |
| G_gauge (local U(1)) | Maximal-tree gauge fixing | Random local gauge transformation |

---

## Theoretical Background

### Normalizing Flows on Quotient Spaces

Given a target distribution π on configuration space X with symmetry group G acting on X, we have an induced distribution π̄ on the quotient space X/G = X/∼ where x ∼ y iff y = g·x for some g ∈ G.

**Key Insight:** If G acts freely and π is G-invariant, then:

$$\pi(x) = \frac{1}{|G|}\bar{\pi}([x])$$

where [x] denotes the equivalence class of x.

**Quotient Flow Strategy:**
1. Train a flow to learn π̄ on X/G (lower-dimensional or less multimodal)
2. Sample from the flow: x̄ ~ q̄_θ
3. Lift to X by applying random g ∈ G: x = g · x̄
4. The lifted distribution q_θ(x) = |G|⁻¹ q̄_θ([x]) should match π(x)

**Potential Advantages:**
- Reduced dimensionality (for translation quotients)
- Removed mode structure (for Z₂ quotients)
- Concentration of probability mass

**Potential Issues (Studied in This Work):**
- Topological obstructions (quotient may have different topology)
- Discontinuous slice maps (canonicalisation not smooth)
- Orbifold singularities (fixed points of G action)

---

## Results and Observables

### Observables Computed

| Observable | Formula | Physical Meaning |
|------------|---------|------------------|
| ⟨φ²⟩ | L⁻ᵈ Σ_x φ_x² | Field fluctuation |
| ⟨φ⁴⟩ | L⁻ᵈ Σ_x φ_x⁴ | Quartic moment |
| ⟨M⟩ | L⁻ᵈ Σ_x φ_x | Magnetisation |
| ⟨\|M\|⟩ | \|L⁻ᵈ Σ_x φ_x\| | Order parameter |
| U₄ | 1 − ⟨M⁴⟩/(3⟨M²⟩²) | Binder cumulant |
| ⟨E_□⟩ | ⟨1 − cos θ_□⟩ | Plaquette energy (gauge) |
| \|P\| | \|L⁻¹ Σ exp(i Σ θ)\| | Polyakov loop (confinement) |

### Metrics

- **Negative Log-Likelihood (NLL):** Training objective; lower indicates better density fit
- **NLL Gap (ΔNLL):** NLL_X − NLL_{X/G}; positive indicates quotient flow advantage
- **Wasserstein-1 Distance:** (Experiment 0 only) Measures sample quality

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quotientflows2024,
  author = {QCD-Internship},
  title = {QuotientFlows: Normalizing Flows on Quotient Spaces for Lattice Field Theory},
  year = {2024},
  url = {https://github.com/QCD-Internship/QuotientFlows}
}
```

---

## License

This project is released for academic research purposes. Please see the repository for licensing details.

---

## Acknowledgements

This work was developed as part of an internship project on generative AI methods for lattice QCD. The experiments are designed to systematically study when and why quotient-space approaches succeed or fail in learning lattice field theory distributions.
