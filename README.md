# Operator Inference and Optimal Control for Power-to-X Methanation

Welcome to the main repository for the modeling and optimal control of a pilot-scale Power-to-X methanation reactor, as presented in our manuscript.

To ensure a clean separation of concerns between data-driven model identification and subsequent control, this project is divided into two distinct submodules. **Each submodule contains its own dedicated `README.md` with detailed instructions on setup, dependencies, and execution.**

---

## 📂 Project Overview & Structure

The repository consists of the following two main directories:

### 1. `OpInf_methanation` (Modeling Framework)
This module contains the Operator Inference (OpInf) framework. It is responsible for learning structured, low-dimensional reduced-order models (ROMs) directly from experimental and synthetic data. To handle measurement noise and partially observed spatial states, it utilizes a neural decoder and a physics-informed (PINN) formulation.
* **Key Tasks:** Data preprocessing, POD basis generation, standard & PINN-OpInf training, and simulation.
* 👉 **[Read the Modeling Instructions here](./OpInf_methanation/README.md)**

### 2. `OpInf_control` (Optimal Control Framework)
Building upon the models derived in the first module, this folder contains the optimal control framework. It leverages the computationally efficient surrogate models to solve nonlinear open-loop optimal control problems, enabling real-time safety verification and process optimization.
* **Key Tasks:** Setting constraints/bounds, solving optimization scenarios, and exploring critical operating regimes.
* 👉 **[Read the Control Instructions here](./OpInf_control/README.md)**

---

## 🚀 Getting Started

Because the modeling and control frameworks have different requirements, they are managed via separate Conda environments (`opinf` and `control`). 

**Recommended Workflow:**
1. Navigate to the `OpInf_methanation` directory first. Follow its README to prepare the data, generate the basis, and train the surrogate models.
2. Once you have trained a model (or if you want to use the provided pre-trained models), navigate to the `OpInf_control` directory. Follow its README to run the optimization scenarios using those models.

---

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.