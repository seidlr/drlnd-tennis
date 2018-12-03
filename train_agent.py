# python train_agent.py --episodes 2000 --model checkpoint --plot Score.png
import argparse
from collections import deque
import datetime
import sys
import time
import os

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ddpg_agent import Agent

def ddpg(n_episodes=2000,
         store_model='checkpoint'):
    """DDPG-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        store_model (str): path for storing pytoch model
    """
    start = time.time()

    scores_all = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations  
        states = np.reshape(states, (1,48))
        agent_0.reset()
        agent_1.reset()
        scores = np.zeros(num_agents)
        while True:
            action_0 = agent_0.act(states, True)           # agent 1 chooses an action
            action_1 = agent_1.act(states, True)           # agent 2 chooses an action
            actions = np.concatenate((action_0, action_1), axis=0) 
            actions = np.reshape(actions, (1, 4))
            env_info = env.step(actions)[brain_name]           # send both agents' actions together to the environment
            next_states = env_info.vector_observations         # get next states
            next_states = np.reshape(next_states, (1, 48))     # combine each agent's state into one state space
            rewards = env_info.rewards                         # get reward
            done = env_info.local_done                         # see if episode finished

            agent_0.step(states, actions, rewards[0], next_states, done, 0) # agent 1 learns
            agent_1.step(states, actions, rewards[1], next_states, done, 1) # agent 2 learns
            scores += rewards                                  # update the score for each agent
            states = next_states                               # roll over states to next time step
            if np.any(done):
                break 
        scores_window.append(np.max(scores))       # save most recent score
        scores_all.append(np.max(scores))              # save most recent score

        if i_episode % 10 == 0:
            print('Episode {}\tMax Reward: {:.3f}\tAverage Reward: {:.3f}'.format(
                i_episode, np.max(scores), np.mean(scores_window)))

        if np.mean(scores_window)>=0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent_0.actor_local.state_dict(), f'{store_model}_actor_0.pth')
            torch.save(agent_0.critic_local.state_dict(), f'{store_model}_critic_0.pth')
            torch.save(agent_1.actor_local.state_dict(), f'{store_model}_actor_1.pth')
            torch.save(agent_1.critic_local.state_dict(), f'{store_model}_critic_1.pth')
            break

    return scores_all

def plot_scores(scores, rolling_window=100, save_plot='Score.png'):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean, linewidth=4)
    plt.savefig('Score.png')
    return rolling_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the agent in the environment')
    parser.add_argument('--episodes', help="How many episodes to train the agent")
    parser.add_argument('--model', default='checkpoint.pth', help="path where the pytorch model should be stored")
    parser.add_argument('--plot', help="path to save the achieved training score of the agent")

    options = parser.parse_args(sys.argv[1:])
    
    env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print("Using {}".format(brain_name))

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # number of actions
    action_size = brain.vector_action_space_size
    print('Size of each actions:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    # print('States look like:', state)
    state_size = states.shape[1]
    print('States have length:', state_size)

    agent_0 = Agent(state_size, action_size, 1, random_seed=0)
    agent_1 = Agent(state_size, action_size, 1, random_seed=0)

    scores = ddpg(n_episodes=int(options.episodes), store_model=options.model)

    plot_scores(scores, rolling_window=10, save_plot=options.plot)
    