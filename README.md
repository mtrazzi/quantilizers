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

## Optional Installation

For Hopper-v2 environments you'll need to install [mujoco](https://github.com/openai/mujoco-py) first.

## Datasets

To import the `Hopper-v2` and `MountainCar-v0` human datasets, run the following command:
```
cd scripts/
sh load_data.sh
cd ..
```

The downloaded datasets will be in `log/[ENV_NAME]/[DATASET_NAME].npz`.

For the Atari Grand Challenge datasets (for Video Pinball), the `v2` datasets are [no longer available](http://atarigrandchallenge.com/data) online.

## Launching the train/test/plot pipeline

Simply run the following command:
```
python quantilizer.py [-h] [--dataset_name DATASET_NAME] [--env_name ENV_NAME]
                      [--do DO [DO ...]] [--seed_min SEED_MIN]
                      [--seed_nb SEED_NB]
                      [--number_trajectories NUMBER_TRAJECTORIES]
                      [--quantiles QUANTILES [QUANTILES ...]]
                      [--render RENDER] [--plotstyle PLOTSTYLE] [--path PATH]
```

where 
- `DATASET_NAME` corresponds to the name of the `.npz` file (for instance "ryan").
- `ENV_NAME` is the name of the gym environment (for instance `Hopper-v2`).
- `DO` is the list of things you want to do for this particular setup (for instance "train" if you just want to train your quantilizer model, or "train test plot" if you want to train the model, generate some rollouts and plot those).
- The trainings, testing, plotting etc. are done for seeds {`SEED_MIN`, `SEED_MIN + 1`, ... , `SEED_MIN + SEED_NB - 1`}.
- `NUMBER_TRAJECTORIES` is the number of episodes generated in rollouts.
- the arguments after `--quantiles` is the list of quantiles you want to do the training, testing, etc. on. For instance `1.0 0.5 0.1`.
- `RENDER` is True or False, and defines whether the `render` method from gym is applied when generating rollouts.
- `PLOTSTYLE` can be `mean_seeds`, `median_seeds` or `distribution`, defining how to aggregate the results from different seeds.
- `PATH` is to save your model in a specific path, for instance can be `run_nb_42/`

Here is a simple example that will generate true & proxy rewards in `log/fig`:
`python quantilizer.py --do train test plot --env_name MountainCar-v0 --quantiles 1.0 0.5 0.25 0.1 0.01 --plotstyle mean_seeds`

# Directory structure

``` bash

#Main files
  |quantilizer.py                          #Main loops (training & rollouts) for the quantilizer algorithm
  
```
