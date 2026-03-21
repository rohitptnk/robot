# Robots: MuJoCo Environments & PPO Training

This repository contains fully configured neural network training and evaluation pipelines for multiple continuous control environments from [Gymnasium (MuJoCo)](https://gymnasium.farama.org/environments/mujoco/) using Proximal Policy Optimization (PPO) via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## 🤖 Supported Environments

Currently, the suite supports the following MuJoCo environments:
- **`half_cheetah/`**: A 2D bipedal robot designed to run forward as fast as possible.
- **`hopper/`**: A one-legged robot that balances and hops forward.
- **`walker2d/`**: A classic bipedal robot that learns to stand and walk.
- **`humanoid/`**: A complex 3D bipedal robot tasked with standing and learning a stable walking gait.

## 🚀 Getting Started

### Prerequisites
It is recommended to run these scripts inside an Anaconda or virtual Python environment. You will need Python 3.8+ and the following specific packages:

```bash
# Install gymnasium with MuJoCo binaries and standard reinforcement learning dependencies
pip install gymnasium[mujoco] stable-baselines3
```

### Training a New Model
To train a model for a specific environment, navigate into its folder and execute the `train.py` script. The training scripts utilize multi-processing (`SubprocVecEnv`) to parallelize environment instances and significantly speed up data collection.

**Example: Training the Humanoid Agent**
```bash
cd humanoid
python train.py
```
*Note for debugging:* To watch the agent attempt its task live during training (which disables multiprocessed environments and relies on a single `DummyVecEnv`), append the render flag.
```bash
python train.py --render
```

During training, intermediate model checkpoints are routinely saved inside a local `models/` directory for each respective environment. 

### Evaluating a Checkpoint
Once you have trained the model (or wish to view an earlier checkpoint), run its corresponding script to load the saved `.zip` from the `models/` directory and run an evaluation loop.

**Example: Evaluating the Humanoid Agent**
```bash
cd humanoid
python evaluate.py
```

## ⚙️ System Configuration & Optimization

All training scripts contain a `NUM_ENVS` variable defining the number of parallel environment states the program will manage simultaneously. 

For the primary machine used in this project, **system hardware specifications and constraints** can be found inside `system_specs.txt`. 

Future contributors or AI Assistants MUST review `system_specs.txt` so `NUM_ENVS` can be appropriately constrained when editing/initiating new model pipelines. Setting this number to match the CPU's available logical cores reduces bottlenecking and guarantees smooth model convergence with optimal performance times!
