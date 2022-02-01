
![mindrake](https://user-images.githubusercontent.com/10000317/150498045-b712992c-b569-4654-a35e-65660df3f795.png)

# Introduction


This repository contains agents for winning the CybORG-CAGE [Challenge(s)](https://github\.com/cage\-challenge/cage\-challenge\-1/tree/main/CybORG).

Our blue agent is developed using a hierarchical reinforcement learning approach which comprises two specialised defensive blue agents and a selection agent that decides the best 'agent for the job' based on a given observation. Our two specialists are trained to defend against the red meander and red b_line agents using the Proximal Policy Optimization (PPO) algorithm.

An important factor in the performance of our specialised agents is that we have allowed them to learn an intrinsic internal reward using curiosity. Curiosity allows our agents to build an internal reward signal which improves performance where rewards from the environment are sparse. Our testing revealed substantial improvements over vanilla PPO where curiosity was applied and also improved upon our attempts to create a more dense reward function manually.

The second important contribution of our solution is a hierarchical approach in which we trained, also using PPO but without curiosity, a selection agent to choose between our specialised agents. We experimented with several different approaches but found best performance using an observation space of one step of the underlying CybORG environment. 

## How to run this code:

```
# List of agents to select from is maintained in agents_list.py
python evalutation.py --name-of-agent <agent>
```

Running hierachical agents
```
# Train sub-agents on agianst the B_linAgent and the RedMeander Agent
# can use either the reward_scaffolded agent or the rllib_alt directories
python train_ppo_cur.py

# add path to weights in the agents/hierachy_agents/sub_agents.py 

#train the rl controller  
train_hier.py

# Run the evaluation script
python evaluation.py
```

Running to create gif of a single episode

```
# create output from an evaluation of a single episode
# alter the config to change the number of steps and the adversary
python get_vis_output.py

cd visualisation

# create gif
python git_maker.py



```



