# OpInf Methanation

This repository contains the Operator Inference (OpInf) framework for modeling and evaluating a pilot-scale Power-to-X methanation reactor, as presented in our manuscript. 

## About the Project
Optimization-based control of complex chemical processes requires models that balance predictive accuracy with rapid evaluability—a requirement that first-principle models often fail to meet. To bridge this gap, we learn structured low-dimensional reduced-order models directly from data using Operator Inference. 

We demonstrate the approach using real-world data from a pilot-scale Power-to-X methanation reactor. Based on experimental sensor measurements, we identify the governing system dynamics under realistic operating conditions. To address challenges arising from measurement noise and partially observed spatial states, we extend the standard Operator Inference framework with a neural decoder for nonlinear field reconstruction and a hybrid physics-informed formulation for robust coefficient identification.

Overall, the results highlight the potential of structured data-driven modeling to bridge physical interpretability and real-time process optimization.

## Repository Structure

```text
OpInf_methanation/
│
├── data/                       # Contains raw data and preprocessing scripts
│   ├── exp_data/               # Experimental reactor data
│   └── smooth_aprbs/           # Synthetic/smoothed APRBS data
│
├── opinf/                      # Core Operator Inference module
│   ├── basis/                  # POD basis generation
│   ├── models/                 # ROM creation and integration
│   ├── training/               # Model training (Standard & PINN-OpInf)
│   ├── pre/ & post/            # Pre-processing and post-processing tools
│   ├── parameters.py           # Centralized configuration file
│   └── ...
│
├── opinf_methanation_exp.py    # Main execution pipeline
├── environment.yml             # Conda environment definition
└── README.md                   # This file


## Dependencies
This project uses a Conda environment to manage dependencies. You can recreate the exact environment required to run the code using the provided `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate opinf

## Getting Started
1. Data Preparation: Before running the main inference script, the raw data must be processed.
Navigate to the data/exp_data and data/smooth_aprbs directories and run the respective data preparation scripts located there. This step extracts, formats, and saves the required .npy files (e.g., flow rate, temperature, time, and inputs).

2. Configuration: All relevant hyperparameters and experimental settings are managed centrally in opinf/parameters.py. By default, the parameters are pre-configured to replicate the first experimental run detailed in the manuscript.

3. Running the Main Pipeline: Once the data is generated and the parameters are set, you can execute the main script from the OpInf_methanation root directory. The script will automatically handle basis generation, lifting, model training, simulation, and plotting.




