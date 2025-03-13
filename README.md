# DeepMimic_Revisited
Recreating DeepMimic using MuJoCo and Gym, with parallel training, experimental model architectures, and flexible motion capture integration 👀

# DeepMimic with MuJoCo and Gym

## Overview

This repository presents a reimplementation of the DeepMimic control framework using MuJoCo and Gym. Our new approach aims to create a flexible, accessible platform for imitation learning—ranging from simple Behaviour Cloning to advanced DeepMimic methods—while supporting parallel training and architecture experimentation. We’re also broadening our dataset support by transitioning from the perfectly formatted Loco_MuJoCo dataset to incorporating CMU's Mocap dataset, enabling more diverse and realistic motion capture inputs.

---

## Original DeepMimic Implementation

### Technical Backbone

- **Custom C++ Simulation:**  
  The original DeepMimic repo featured a custom-built C++ simulation and visualization environment. This engine was designed to simulate high-dimensional character dynamics and integrate motion capture data into a physics-based control policy.

- **Dataset and Data Formatting:**  
  The dataset consisted of high-dimensional states (positions, orientations, joint angles, velocities, and phase variables) with corresponding actions (target joint configurations or torques), maintaining a strict 1:1 mapping between inputs and outputs for realistic imitation.

- **Learning Pipeline:**  
  The original system employed a combination of imitation learning and later reinforcement learning, with loss functions tuned to capture multiple aspects of motion fidelity.

### Challenges

- **Complexity:**  
  The powerful, custom C++ environment proved challenging to extend and modify, slowing rapid prototyping and experimentation.

- **Accessibility:**  
  The custom tools and visualization systems limited contributions and hindered flexible experimentation with alternative datasets or learning strategies.

---

## Our Implementation: DeepMimic with MuJoCo and Gym

### The Proposed Plan

1. **MuJoCo-based Physics Simulation:**  
   We now use MuJoCo for its robust and reliable physics simulation, combined with the Gym environment for a standardized interface. This setup supports both simple imitation learning and the more complex DeepMimic approach.

2. **Dataset Flexibility:**  
   Our experiments began with the Loco_MuJoCo dataset to build familiarity through Behaviour Cloning. Moving forward, we plan to integrate CMU's Mocap dataset to offer more varied and realistic motion capture data, overcoming the limitations of a perfectly formatted dataset.

3. **Training Pipeline in PyTorch:**  
   The control policy is implemented in PyTorch. We support both a Behaviour Cloning pipeline—where initial results have been impressive (as demonstrated by our gif)—and a full DeepMimic imitation learning approach, which is in early stages and undergoing refinement.

4. **Parallel Training and Architecture Exploration:**  
   The implementation is designed to support parallel training, accelerating experimentation. We’re also exploring modifications to the model architecture to improve performance and better capture complex character dynamics.

5. **Modular and Configurable Design:**  
   Hyperparameters, including training epochs, learning rate, simulation steps, and model dimensions, are fully configurable via `config.yaml`, ensuring that our system remains flexible and easy to extend.

6. **Simulation and Evaluation:**  
   The simulation module (`simulate.py`), now integrated with MuJoCo and Gym (see `mujoco_env.py`), renders high-fidelity simulations that allow for thorough evaluation of our trained models.

---

## Current Status and Future Work

- **Behaviour Cloning:**  
  We have achieved strong results with our Behaviour Cloning approach, and a demonstration gif is included in the repository to showcase these achievements.

![Imitation Learning Demo](https://github.com/user-attachments/assets/ed0af0c8-39b8-4e09-9a14-e324587de846)

- **DeepMimic Implementation:**  
  Our initial DeepMimic results are preliminary and not yet on par with our Behaviour Cloning performance. Ongoing work is focused on refining these results by:
  - Optimizing the training pipeline,
  - Testing novel model architectures,
  - Enhancing parallel training strategies, and
  - Integrating the CMU Mocap dataset for richer motion data.

![Initial DeepMimic Result](https://github.com/user-attachments/assets/791cef3e-2439-4d88-9604-939077b11d98)


Our roadmap is dedicated to overcoming the current challenges and pushing towards more robust and realistic control policies.
