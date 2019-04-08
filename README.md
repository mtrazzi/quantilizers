Let's play the game `Hopper-v2` ([gym](https://gym.openai.com/envs/Hopper-v2/); [source](https://gym.openai.com/envs/Hopper-v2/)).

# Getting paid

See the rules [here](https://docs.google.com/document/d/1Xplmvf_UmAzsbg10ATrTYc-qsiVdiUjtUwXsqZLyrZM/edit?usp=sharing).

# Getting Started

1) ```git clone https://github.com/mtrazzi/quantilizers.git```

2) ```cd quantilizers```

2) ```pip install -Ur gather_data_requirements.txt```

3) ```python gather_data.py```

4) Follow the specific instructions for your game (see below).

# Instructions for Hopper-v2

1) After launching ```python gather_data.py```, you should see some mujocopy rendering (see image below).

2) Press `d` to have "render every frame" set to `Off`, and press `Tab` to have an horizontal camera angle that follows you.

2) After step 1), you will now start playing the game using your keyboard. To allow PyGame to listen to your keyboard movements, **you must have the focus on the pygame black window** (i.e. the black window must be in front of any rendering from the environment).

![black window pygame](doc/img/black_window_pygame.png)

3) When the pygame black window is in front, the keys to move are {j,k} (for the foot), {w,s} for the top junction and {a,d} for the middle junction. Those keys can be directly modified in the [`gather_data.py`](https://github.com/mtrazzi/quantilizers/blob/master/gather_data.py) file, replacing {j,k,w,s,a,d} with your keys.

4) To do a few steps, you can mostly use the keys for the foot (jk) and sometimes use the keys from the upper junction (sw). It's recommended to watch a video of a trained AI performing the Hopper-v2 task [here](https://www.youtube.com/watch?v=2lf-3tgWiUc&t=0m45s) before playing.
