import gym
import pygame
import sys
import time
import matplotlib
import numpy as np
import os
import os.path as osp
import pyglet.window as pw
from collections import deque
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from threading import Thread

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

def play_1d(env, fps=30, zoom=None, callback=None, keys_to_action=None, save=False, filename='traj.npz'):

    running_reward = None
    counter = 1
    obs_s = env.observation_space
    assert type(obs_s) == gym.spaces.box.Box

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))
    pressed_keys = []
    running = True
    env_done = True
    #screen requires arg of length 2 (restricted to 2d games)
    #video_size = env.observation_space.shape[0], env.observation_space[1]
    screen = pygame.display.set_mode((300,300))
    clock = pygame.time.Clock()

    if save:
        data_save = {'obs':[], 'acs':[], 'rews':[], 'done':[]}

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
            if data_save['rews']:
                reward_sum = np.sum(data_save['rews'][-1])
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print("### TRIAL #{} ###".format(counter))
                print("reward this trial: ", reward_sum)
                print("running reward is: ", running_reward)
                print()
                counter += 1
            if save:
                for key in data_save.keys():
                    data_save[key].append([])
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)
            if save:
                data_save['obs'][-1].append(prev_obs)
                if action is 0: #for mujoco: catch empty actions
                    action = np.zeros(env.action_space.shape)
                data_save['acs'][-1].append(action)
                data_save['rews'][-1].append(rew)
                data_save['done'][-1].append(False)
            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)
        if obs is not None:
            if len(obs.shape) == 2:
                obs = obs[:, :, None]
            env.render()
        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)

    if save:
        def pad1d(arr, l, val=0.):
            out = np.full((l,), val)
            out[:len(arr)] = arr
            return out
        def pad2d(arr, shape, val=0.):
            #pad variable second dimension
            assert len(shape) == 2
            out = np.full(shape, val)
            out = np.array([pad1d(row, shape[1], val=val) for row in arr])
            return out
        def pad3d(arr, shape, val=0.):
            #pad variable second dimension
            out = np.full(shape, val)
            for i, mat in enumerate(arr):
                out[i,:len(mat),:] = np.array(mat)
            return out

        #pad obs to max ep length. maybe should edit this to have ep length as an argument
        n_eps = len(data_save['obs'])
        ep_length = max([len(i) for i in data_save['obs']])
        obs_arr = np.zeros((n_eps, ep_length, obs_s.shape[0]))
        for i, obs in enumerate(data_save['obs']):
            obs_arr[i,:len(obs),:] = obs
        data_save['obs'] = obs_arr
        data_save['done'] = pad2d(data_save['done'], (n_eps, ep_length), val=True)
        data_save['ep_rets'] = np.array([np.sum(rews) for rews in data_save['rews']])
        data_save['rews'] = pad2d(data_save['rews'], (n_eps, ep_length))
        data_save['acs'] = pad3d(data_save['acs'], (n_eps, ep_length, len(data_save['acs'][0][0]))) #for mujoco
        directory = osp.dirname(filename) 
        if not osp.exists(directory):
            os.makedirs(directory)
        np.savez(filename, **data_save)

    pygame.quit()
