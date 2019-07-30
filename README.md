# Quantilizers

[How useful is quantilization for mitigating specification-gaming?](https://drive.google.com/uc?export=download&id=13qAfOm8McRvXS33MCNH0ia4ApMIClZP9) introduces variants of several classic environments (Mountain Car, Hopper and Video Pinball) where the observed reward differs from the true reward, creating an opportunity for the agent to game the specification of the observed reward. The paper shows that a quantilizing agent avoids specification gaming and performs better in terms of true reward than both imitation learning and a regular RL agent on all the environments. This repository contains the code to reproduce the experiments from the paper.

# Getting Started

## Installation

Clone this repository and install the other dependencies with `pip`:

```
git clone https://github.com/mtrazzi/quantilizers.git
cd quantilizers
pip install -U -r requirements.txt
```

## Datasets

To import the `Hopper-v2` and `MountainCar-v0` human datasets, run the following command:
```
cd scripts/
sh load_data.sh
cd ..
```

The downloaded datasets will be in `log/[ENV_NAME]/[DATASET_NAME].npz`.

For the Atari Grand Challenge datasets (for Video Pinball), the `v2` datasets are [no longer available](http://atarigrandchallenge.com/data) online.

# Files overview

``` bash

#Main files
  |quantilizer.py                          #Main loops (training & rollouts) for the quantilizer algorithm
  
```
