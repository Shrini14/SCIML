# FNO – Poisson 2D (Synthetic Experiment)

This folder contains a minimal end-to-end experiment using a 2D Fourier Neural Operator (FNO)
to learn the solution operator of the Poisson equation.

## Problem
-Δu(x,y) = f(x,y) on [0,1]²

Synthetic forcing functions are generated, and analytical solutions are used
to validate the FNO pipeline.

## Dataset
- Grid: 64 × 64
- Total samples: 8
- Train: 6
- Test: 2
- Format: (B, 1, H, W)

## Run
```bash
python experiments/train_poisson_2d.py
