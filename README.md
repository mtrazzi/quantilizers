[How useful is quantilization for mitigating specification-gaming?](https://drive.google.com/uc?export=download&id=13qAfOm8McRvXS33MCNH0ia4ApMIClZP9) introduces variants of several classic environments (Mountain Car, Hopper and Video Pinball) where the observed reward differs from the true reward, creating an opportunity for the agent to game the specification of the observed reward. The paper shows that a quantilizing agent avoids specification gaming and performs better in terms of true reward than both imitation learning and a regular RL agent on all the environments. This repository contains the code to reproduce the experiments from the paper.

## Files overview

``` bash

#Main files
  |quantilizer.py                          #Main loops (training & rollouts) for the quantilizer algorithm
  
```
