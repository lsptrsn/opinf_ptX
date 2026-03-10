# OpInf Control

This repository contains the optimal control framework for the pilot-scale Power-to-X methanation reactor. Building upon the models derived via Operator Inference (OpInf), this module focuses on utilizing these computationally efficient surrogate models to solve nonlinear open-loop optimal control problems. 

As highlighted in our manuscript, this framework enables the exploration of critical operating regimes and demonstrates the feasibility of real-time safety verification and process optimization.

## Repository Structure

```text
OpInf_control/
│
├── config/                     # Configuration files for the control setup
├── data/                       # Datasets required for the control scenarios
├── models/                     # Saved OpInf models (e.g., for synthetic data)
├── results/                    # Output directory for optimization results and plots
├── utils/                      # Helper functions and utilities for the control problem
│
├── optimal_control_main.py     # Main execution script for the optimal control problem
├── environment.yml             # Conda environment definition
└── README.md                   # This file


## Dependencies
This project uses a Conda environment to manage dependencies. You can recreate the exact environment required to run the code using the provided `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate control

## Getting Started

### 1. Pre-trained Models
The default OpInf model trained on the noise-free synthetic dataset is already saved in the `models/` directory. 
* **Replacing the model:** If you wish to use a different model (e.g., one you trained yourself in the `OpInf_methanation` module), simply place your saved model file into the `models/` directory. You will then need to update the model file name reference in the configuration settings.

### 2. Configuration
The parameters for the optimal control problem (such as constraints, bounds, objective function weights, and model selection) can be adjusted in two places:
* Inside the `config/` directory files.
* Directly within the main script `optimal_control_main.py`.

### 3. Running the Optimization
Once your environment is set up and your configuration is to your liking, execute the main script to solve the open-loop optimal control problem:

```bash
python optimal_control_main.py



