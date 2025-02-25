# DeepMimic_Revisited
Reimplementing DeepMimic with GPU acceleration on a more modern simulation suite with a few added improvements 👀

# DeepMimic with IsaacGym

## Overview

This repository presents a modern reimplementation of the DeepMimic control framework using PyTorch and NVIDIA's IsaacGym as the simulation backend. By leveraging IsaacGym, we can run GPU-accelerated physics simulations, significantly speeding up the training and evaluation of our control policies.

Our goal is to reproduce the architecture and core ideas of DeepMimic—using expert motion capture data to train a control policy—while replacing the original C++ simulation and the alterniatve Mujoco implementation with a GPU-based physics simulator.

---

## Original DeepMimic Implementation

### Technical Backbone

- **Custom C++ Simulation:**  
  The original DeepMimic repo featured a custom-built C++ simulation and visualization environment. This engine was designed to simulate high-dimensional character dynamics, integrating motion capture data into a physics-based control policy.

- **Dataset and Data Formatting:**  
  The dataset consists of high-dimensional states (including positions, orientations, joint angles, velocities, and phase variables) and corresponding actions (target joint configurations or torques). This strict 1:1 mapping between data inputs and simulation outputs was essential for achieving realistic imitation.

- **Learning Pipeline:**  
  The system used a combination of imitation learning (and later reinforcement learning) to train the control policy, employing loss functions that accounted for multiple aspects of motion fidelity.

### Challenges

- **Complexity:**  
  The original C++ environment was powerful but complex and difficult to extend, especially for rapid prototyping.
  
- **Accessibility:**  
  The custom tools and visualization systems posed a barrier to new contributors and limited experimentation.

---

## Our Implementation: DeepMimic with IsaacGym

### The Proposed Plan

1. **Leveraging GPU-Based Physics with IsaacGym:**  
   Instead of a custom C++ simulation or Mujoco, we use IsaacGym to run physics simulations entirely on the GPU. IsaacGym’s Python API lets us simulate complex character dynamics with high performance and scalability.

2. **Dataset Compatibility:**  
   We will use the same dataset as the original DeepMimic, formatted identically. A dedicated `data_loader.py` module is provided to parse and prepare the data, ensuring a 1:1 mapping between the original inputs/outputs and our simulation environment.

3. **Training Pipeline in PyTorch:**  
   The control policy is implemented in PyTorch, mapping high-dimensional state inputs (as defined in the dataset) to action outputs. The training pipeline uses imitation learning to reproduce the expert motions.

4. **Modular and Configurable Design:**  
   Hyperparameters (such as training epochs, learning rate, simulation steps, and model dimensions) are configurable via `config.yaml`, ensuring that our implementation is flexible and easy to extend.

5. **Simulation and Evaluation:**  
   The simulation module (`simulate.py`), now integrated with IsaacGym (see `isaac_env.py`), runs the trained model, rendering high-fidelity simulations using GPU-accelerated physics.

### Our Advantages

- **High-Performance Simulation:**  
  With IsaacGym, we harness the power of GPU-based physics simulation, making training faster and enabling real-time simulation of complex dynamics.

- **Ease of Use and Extension:**  
  By using a modern Python-based framework and a well-supported simulator, our implementation is more accessible and easier to modify than the original C++ codebase.

- **Reproducibility and Collaboration:**  
  The repository is structured to make it easy for new contributors to understand the technical backbone of DeepMimic and the improvements we’ve introduced using IsaacGym.

---
