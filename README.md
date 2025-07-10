# Polydisperse Sphere Scattering Analysis

A computational framework for analyzing scattering intensity I(Q) from polydisperse sphere systems using molecular dynamics simulations, theoretical models, and machine learning approaches.

## Project Overview

This project investigates the structure factor and scattering properties of polydisperse hard sphere systems with different size distributions (uniform, normal, and log-normal). The analysis combines:

- **Molecular Dynamics Simulations**: LAMMPS-based simulations of polydisperse sphere systems
- **Scattering Calculations**: Computation of structure factors I(Q) from particle configurations
- **Theoretical Models**: Percus-Yevick (PY) theory predictions with beta corrections
- **Machine Learning**: Variational Autoencoders (VAE) for generative modeling and parameter inference

## Repository Structure

```
├── code/                    # Core computation and simulation
│   ├── calc_Iq.cpp         # C++ implementation for I(Q) calculation
│   ├── calc_Iq.py          # Python implementation for I(Q) calculation
│   ├── *.lammps            # LAMMPS input scripts for different distributions
│   └── run_lmp.sh          # Simulation execution scripts
├── analyze/                 # Analysis and modeling
│   ├── analyze.py          # Core analysis functions and PY theory
│   ├── ML_analyze.py       # Machine learning data processing
│   ├── VAE_model.py        # Neural network models (VAE, Generator, Inferrer)
│   └── main_*.py           # Main analysis scripts
├── plot/                   # Visualization and plotting
│   ├── illustrate_plot.py  # Basic visualization functions
│   ├── NN_model_plot.py    # Neural network performance plots
│   └── SVD_plot.py         # Singular value decomposition analysis
├── data/                   # Simulation data (archived)
├── data_used/              # Processed datasets
└── ref/                    # Reference materials
```

## Key Features

### 1. Scattering Intensity Calculation
- **C++ Implementation**: High-performance calculation using spherical averaging
- **Python Implementation**: Flexible analysis with scipy integration
- Both implementations compute I(Q) from particle positions and diameters

### 2. Polydispersity Models
Three size distribution types are supported:
- **Type 1**: Uniform distribution
- **Type 2**: Normal (Gaussian) distribution
- **Type 3**: Log-normal distribution

### 3. Theoretical Comparisons
- **Percus-Yevick Theory**: Classical liquid state theory predictions
- **Beta Corrections**: Enhanced PY theory for better accuracy
- **RMSE Analysis**: Quantitative comparison between simulation and theory

### 4. Machine Learning Framework
- **Variational Autoencoder (VAE)**: Learns latent representations of I(Q) curves
- **Generator Model**: Generates I(Q) from physical parameters (η, σ)
- **Inferrer Model**: Predicts physical parameters from I(Q) measurements
- **SVD Analysis**: Dimensionality reduction and feature extraction

## Usage

### Running Simulations
```bash
# Execute LAMMPS simulation
cd code/
./run_lmp.sh

# Calculate I(Q) from simulation output
./calc_Iq prec 18 1 0.30 0.10 ../data/
```

### Analysis Workflow
```python
# Basic analysis
from analyze.analyze import *

# Load and smooth I(Q) data
q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file(folder, finfo)

# Compare with PY theory
compare_Iq_by_param(folder, params, label="comparison")
```

### Machine Learning Training
```python
# Train VAE models
from analyze.VAE_model import *

# Prepare data and train models
train_and_save_VAE_alone(folder, label, latent_dim=3)
train_and_save_generator(folder, label, vae_path, input_dim=2)
train_and_save_inferrer(folder, label, vae_path, output_dim=2)
```

## Dependencies

### Core Requirements
- **C++**: Standard library with filesystem support
- **Python 3.8+**: NumPy, SciPy, Matplotlib
- **PyTorch**: For neural network models
- **scikit-learn**: For Gaussian process regression

### Simulation Requirements
- **LAMMPS**: Molecular dynamics simulation package

### Analysis Libraries
```bash
pip install numpy scipy matplotlib torch scikit-learn
```