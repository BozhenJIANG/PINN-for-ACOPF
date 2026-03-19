# PINN-for-ACOPF

## Overview

This repository contains the simplified implementation accompanying the paper:  
**"Unsupervised Online Learning for AC Optimal Power Flow: A Gradient-Guided Physics-Informed Neural Network Approach"**.

Due to GitHub's 25 MB file size limit, the dataset is not included in this repository. However, users can generate their own data by running the `Data_Generator.jl` script.

## Repository Contents

- **`Data_Generator.jl`** – Generates the training and test datasets for the AC Optimal Power Flow (ACOPF) problem.  
- **`Distribution_Display.py`** – Visualizes the distribution of the generated data.  
- **`Check_ACPF_Balance.py`** – Verifies the correctness of the generated data by checking power flow balance.  
- **`ACOPF_Solver-torch.py`** – Main training and evaluation script. Running this will produce results for the IEEE 118-bus system similar to those reported in the paper.  
  > **Note:** The data distribution used in this code may differ slightly from the original paper, so the obtained ACOPF cost might vary marginally.

## Simplified Implementation

For clarity, we have intentionally simplified the network architecture and training procedure. The code does **not** employ tensor computation optimizations, making the gradient calculation process more transparent and easier to follow for readers.

## Requirements

The code is written in Python (with PyTorch) and Julia (for data generation). Please ensure you have the necessary dependencies installed (e.g., PyTorch, NumPy, Matplotlib, and Julia with appropriate packages).
