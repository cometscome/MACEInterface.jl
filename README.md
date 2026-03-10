# MACEInterface.jl

[![Build Status](https://github.com/cometscome/MACEInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/MACEInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia interface for the [MACE](https://github.com/ACEsuit/mace)
machine-learning interatomic potential.

**MACEInterface.jl** provides a lightweight Julia wrapper around the
official Python implementation of **MACE (Message Passing Atomic Cluster
Expansion)** so that trained MACE models can be evaluated directly from
Julia.

The package focuses on **energy, force, stress, and virial evaluation**
and is intended to integrate MACE potentials into Julia simulation
workflows.

Python dependencies are managed automatically using `CondaPkg.jl`.

------------------------------------------------------------------------

# Important

This package is **not part of the official MACE project**.

All credit for the MACE method and its implementation goes to the
original authors and developers.

Original MACE repository:

https://github.com/ACEsuit/mace

If you use MACE in scientific work, please cite the original
publications.

------------------------------------------------------------------------

# What is MACE?

**MACE (Message Passing Atomic Cluster Expansion)** is a machine
learning interatomic potential based on **E(3)-equivariant graph neural
networks**.

It enables highly accurate predictions of atomistic properties such as

-   energies
-   forces
-   stresses
-   virials

for molecules and materials.

Project repository:

https://github.com/ACEsuit/mace

Key reference:

Batatia et al.\
*MACE: Higher Order Equivariant Message Passing Neural Networks for Fast
and Accurate Force Fields*\
NeurIPS 2022

https://arxiv.org/abs/2206.07697

------------------------------------------------------------------------

# Installation

``` julia
using Pkg
Pkg.add(url="https://github.com/cometscome/MACEInterface.jl")
```

Python dependencies such as

-   mace-torch
-   ase
-   torch
-   numpy

are installed automatically using `CondaPkg.jl`.

------------------------------------------------------------------------

# Quick Example

``` julia
using MACEInterface

symbols = ["O","H","H"]

positions = [
0.000000 0.000000 0.000000
0.758602 0.000000 0.504284
-0.758602 0.000000 0.504284
]

cell = [
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
]

pot = MACEPotential(
"model.model",
symbols,
positions;
cell=cell,
pbc=(true,true,true),
device="cpu"
)

energy(pot)
forces(pot)
stress(pot)
virial(pot)
```

------------------------------------------------------------------------

# Updating structures

Atomic positions and simulation cell can be updated without recreating
the calculator.

``` julia
set_positions!(pot, new_positions)

set_cell!(pot, new_cell)

set_pbc!(pot, (true,true,true))
```

------------------------------------------------------------------------

# Combined evaluation

Convenience functions are provided for common evaluation patterns.

``` julia
e, f = energy_forces(pot, positions)

e, f, σ = energy_forces_stress(pot, positions)

e, f, W = energy_forces_virial(pot, positions)
```

------------------------------------------------------------------------

# Units

The interface follows the standard ASE / MACE unit conventions.

  quantity   unit
  ---------- ---------
  energy     eV
  length     Å
  force      eV / Å
  stress     eV / Å³
  virial     eV

You can query them programmatically:

``` julia
unit_system()
```

------------------------------------------------------------------------

# Python backend

This package internally calls the official Python MACE implementation
using `PythonCall.jl`.

Python dependencies are automatically managed with `CondaPkg.jl`, so no
manual Python installation is required.

------------------------------------------------------------------------

# Limitations

Current limitations include:

-   atomic virials are not exposed
-   training workflows are not supported (use the Python MACE tools)
-   GPU support depends on the Python backend configuration

------------------------------------------------------------------------

# Related software

-   MACE\
    https://github.com/ACEsuit/mace

-   ASE (Atomic Simulation Environment)

-   PyTorch

------------------------------------------------------------------------

# License

MIT

------------------------------------------------------------------------

# Acknowledgement

This package would not exist without the excellent work of the MACE
developers.

If you use MACE, please consider supporting the original project:

https://github.com/ACEsuit/mace
