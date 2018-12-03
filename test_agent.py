# python test_agent.py --actor_0_model checkpoint_actor_0.pth --critic_0_model checkpoint_critic_0.pth --actor_1_model checkpoint_actor_1.pth --critic_1_model checkpoint_critic_1.pth
import argparse
import sys
import os

from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
import torch

from ddpg_agent import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the agent in the environment')
    parser.add_argument('--actor_0_model', required=True, help="path to the saved pytorch first actor model")
    parser.add_argument('--critic_0_model', required=True, help="path to the saved pytorch first critic model")
    parser.add_argument('--actor_1_model', required=True, help="path to the saved pytorch second actor model")
    parser.add_argument('--critic_1_model', required=True, help="path to the saved pytorch second critic model")

    result = parser.parse_args(sys.argv[1:])

    print (f"Selected first actor model {result.actor_0_model}")
    
    if os.path.isfile(result.actor_0_model):
        print ("First Actor model exists")
    else: 
        print ("First Actor model not found")

    print (f"Selected critic model {result.critic_0_model}")
    if os.path.isfile(result.critic_0_model):
        print ("First Critic model exists")
    else: 
        print ("First Critic model not found")

    print (f"Selected second actor model {result.actor_1_model}")
    
    if os.path.isfile(result.actor_1_model):
        print ("Second Actor model exists")
    else: 
        print ("Second Actor model not found")

    print (f"Selected critic model {result.critic_1_model}")
    if os.path.isfile(result.critic_1_model):
        print ("Second Critic model exists")
    else: 
        print ("Second Critic model not found")        

    
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

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

    agent_0 = Agent(state_size=state_size, 
                    action_size=action_size, 
                    num_agents=1, 
                    random_seed=0)

    agent_1 = Agent(state_size=state_size, 
                  action_size=action_size, 
                  num_agents=1, 
                  random_seed=0)                    

    agent_0.actor_local.load_state_dict(torch.load(result.actor_0_model, map_location='cpu'))
    agent_0.critic_local.load_state_dict(torch.load(result.critic_0_model, map_location='cpu'))
    agent_1.actor_local.load_state_dict(torch.load(result.actor_1_model, map_location='cpu'))
    agent_1.critic_local.load_state_dict(torch.load(result.critic_1_model, map_location='cpu'))

    # Set environment to evalulation mode
    env_info = env.reset(train_mode=False)[brain_name]        
    states = env_info.vector_observations                  
    states = np.reshape(states, (1,48))
    scores = np.zeros(num_agents) 

    for i in range(200):
        action_0 = agent_0.act(states, add_noise=False)         
        action_1 = agent_1.act(states, add_noise=False)        
        actions = np.concatenate((action_0, action_1), axis=0) 
        actions = np.reshape(actions, (1, 4))
        env_info = env.step(actions)[brain_name]        
        next_states = env_info.vector_observations        
        next_states = np.reshape(next_states, (1, 48))
        rewards = env_info.rewards  
        scores += rewards                        
        dones = env_info.local_done                 
        states = next_states                              
        if np.any(dones):                              
            break

    print('Score (max over agents) from episode {}: {}'.format(i+1, np.max(scores)))            