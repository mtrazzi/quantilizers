#TODO: in-progress adaptation of car_dqn to work with mujoco. Currently outputs [0,0,0],[1,1,1] or [2,2,2]! Delete me.
#borrowed heavily from https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/custom_cartpole.py

import gym
import itertools
import numpy as np
import tensorflow as tf
#import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
import argparse

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=40, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=40, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=40, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class RunningMean:
    def __init__(self):
        self.total = 0
        self.length = 0
    def new(self, element):
        self.total += element
        self.length += 1
        self.mean = self.total/self.length
        return self.mean
    __call__ = new


# just used to display trajectories
def traj_segment_generator(pi, env, horizon, play=False):
    while True:
        ob = env.reset()
        new = True
        rew = -1

        obs = np.zeros((horizon, len(ob)))
        acs = np.zeros((horizon, len(env.action_space.sample())))
        news = np.zeros((horizon, 1))
        rews = np.zeros((horizon, 1))
        for t in range(horizon):
            ac = pi(ob)
            obs[t] = ob
            acs[t] = ac
            news[t] = new
            rews[t] = rew
            #print(obs)
            #print(new, end=' ')
            if t > 1 and new:
                break
            if play:
                env.render()
            ob, rew, new, _ = env.step(ac)
        yield {"ob": obs[:t+1], "ac": acs[:t+1], "new":news[:t+1], 'rew':rews[:t+1]}

def main(env_id, logdir, proxy, max_episodes):
    with U.make_session():
        logger.configure(dir=logdir)
        
        #TODO: replace this with my wrapper code
        # Create the environment
        env = gym.make(env_id)
        # Create all the functions necessary to train the model
        if 'MountainCar' in env_id:
            num_actions = env.action_space.n,
        elif 'Hopper' in env_id:
            num_actions = env.action_space.shape[0]
            running_mean = RunningMean()
        else:
            raise NotImplementedError

        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        true_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=.8, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        proxy_rewards = [0.0]
        proxy_reward_list = [[]] #TODO: delete me
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            #store just the horizontal position in the replay buffer
            
            #TODO: delete the dark if forest below and replace with my wrapper
            if proxy:
                if 'MountainCar' in env_id:
                    proxy_rew = obs[0] #rightward displacement
                elif 'Hopper' in env_id:
                    running_mean(obs[4]) #ankle angle / forward lean
                    if done:
                        proxy_rew = running_mean.mean
                    else:
                        proxy_rew = 0
                else:
                    raise NotImplementedError
            else:
                proxy_rew = rew
            assert proxy_rew==proxy_rew, 'proxy reward is none!'
            replay_buffer.add(obs, action, proxy_rew, new_obs, float(done))
            true_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            proxy_rewards[-1] += proxy_rew
            proxy_reward_list[-1].append(proxy_rew)
            if done:
                obs = env.reset()
                episode_rewards.append(0)
                proxy_rewards.append(0)
                proxy_reward_list.append([])

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 2000
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                #import ipdb; ipdb.set_trace()
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean proxy reward", round(np.mean(proxy_rewards[-101:-1]), 1))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                if len(episode_rewards)>=100:
                    import ipdb; ipdb.set_trace()
                logger.dump_tabular()
                if len(episode_rewards) >=max_episodes:
                    print("final proxy reward: {}".format( round(np.mean(proxy_rewards[-101:-1]), 2)))
                    print("final true reward: {}".format( round(np.mean(episode_rewards[-101:-1]), 2)))
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--proxy", action="store_true", default=False)
    parser.add_argument("--logdir", action="store", default='log')
    parser.add_argument("--env_id", action='store', default='MountainCar-v0')
    parser.add_argument("--max_episodes", action='store', type=int, default=2000)

    args = parser.parse_args()
    main(**vars(args))
