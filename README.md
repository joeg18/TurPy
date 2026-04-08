# TurPy: Differentiable Wave Optics Turbulence Simulator

TurPy is an open-source, GPU-accelerated wave optics simulator designed for high-fidelity turbulence modeling and end-to-end optical system design. With its fully differentiable architecture, TurPy bridges the gap between turbulence simulation and gradient-based optimization, supporting applications such as synthetic data generation, turbulence-informed algorithm development, and optical platform optimization under turbulent conditions. Built natively in PyTorch, TurPy exposes the full split-step propagation pipeline to automatic differentiation, enabling:

- **Synthetic data generation** for training and evaluating algorithms under representative turbulence conditions
- **End-to-end gradient-based optimization** of optical elements, including diffractive neural networks and adaptive optics architectures
- **Physics-accurate simulation** validated against established 2nd and 4th order atmospheric statistics across weak to strong turbulence regimes


## Key Features

### Split-Step Wave Optics Propagation
TurPy implements a split-step propagation model that alternates between an angular spectrum diffraction step and a real-space phase screen modulation step. The angular spectrum transfer function is used in place of the Fresnel approximation, accurately capturing wide-angle scattering in strongly turbulent paths without paraxial error.

### Subharmonic Phase Screen Generation
Computational efficiency demands smaller simulation grids, but this implicitly limits the largest turbulent scale represented in a phase screen. TurPy incorporates a subharmonic generation method that recursively injects low-frequency PSD content into sub-grids near the DC coordinate increase in the low-frequency regime without inflating array size. This accurately captures outer-scale effects, including low-order tip-tilt contributions, that are missed by nominally-sampled screens.

### Autoregressive Temporal Evolution
TurPy models the temporal evolution of turbulence through two independent mechanisms: Fourier-domain phase tilt to capture wind-driven translation, and PSD-weighted stochastic injection to model turbulent boiling independent of advection. A user-calibrated decorrelation parameter `α` relates the blending weight between successive realizations to a physical decorrelation time, allowing TurPy to reproduce realistic Greenwood frequency dynamics.

### Automated Phase Screen Placement
Split-step accuracy requires two constraints to be simultaneously satisfied: Fourier aliasing limits set by the angular spectrum transfer function, and weak-turbulence approximations requiring the Rytov variance per interval to remain below unity. TurPy's automated placement routine takes a user-defined Cn² profile and iteratively splits, merges, and refines screen intervals until both constraints are met across the full propagation path — eliminating the need for manual screen placement and ensuring physical validity at any turbulence strength.

### Differentiable Architecture
TurPy is implemented entirely in PyTorch, using `torch.fft`, complex tensor operations, and standard autograd-compatible functions throughout. The full forward propagation pipeline — from phase screen generation through field propagation — supports reverse-mode automatic differentiation, enabling gradients to flow directly through simulated turbulence into any upstream optical or algorithmic parameters being optimized.

### GPU Acceleration
All operations are tensor-native and device-agnostic. Simulations run on CPU or GPU with no code changes. Large-batch synthetic data generation and gradient-based optimization both benefit from GPU acceleration, with training runs for diffractive architectures completing in under one hour on consumer hardware.

### Multi-Domain PSD Support
Phase screen statistics are derived from a user-specified power spectral density following media-specific power laws. The modified Von Karman PSD is provided for atmospheric turbulence (validated in cited work), with documented hooks for oceanic and biological turbulence descriptions following characterized power laws. Any medium for which a PSD can be defined is supported with minimal code changes.


---
## Core Functionalities
# Simulation Modules 

- **StaticPhaseScreen**
Simulates phase screens based on static statistics with optional subharmonic injections.

- **TemporalPhaseScreen**
Models time-dependent turbulence using autoregressive updates and frozen flow with wind velocity vectors.

- **LogAmplitudePhaseScreen**
Supports strong turbulence simulations with combined amplitude and phase distortions. Note: Prototype functionality non calibrated nor demonstrated in paper.

- **Propagation Kernels**
Coherent propagation in 1D and 2D models including:

  - Pre-built angular spectrum propagators.
  - Split-step implementation for turbulence-adaptive field propagation.
  - Power Spectral Density (PSD) Models
  - Kolmogorov: Classical turbulence model.
  - Modified von Karman: Includes inner and outer scales (l₀, L₀).
  - Custom PSD: User-defined models for unique turbulence scenarios.
  - Turbulence Path Sampling
  - Generates step sizes and rolling Fried parameters (r₀).
  - Adaptive sampling minimizes Fourier aliasing and adheres to weak-turbulence assumptions.
  - Boundary Handling
  - Absorbing boundaries reduce edge effects during wavefront propagation.

---
## Example Usage
After installation, execute the turpy_unit_test.ipynb file below for demonstration of core functionality & simulation.


## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/your-repo/TurPy.git
cd TurPy
pip install -r requirements.txt
```

## Citation
> **Associated Publication:** J. L. Greene, A. Moore, I. Ochoa, E. Kwan, P. Marano, and C. R. Valenta, "TurPy: a physics-based and differentiable optical turbulence simulator for algorithmic development and system optimization," *Defense and Sensing. Synthetic Data for Artificial Intelligence and Machine Learning: Tools, Techniques and Applications IV*, SPIE, 2026.
