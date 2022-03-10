
![mindrake](https://user-images.githubusercontent.com/10000317/150498045-b712992c-b569-4654-a35e-65660df3f795.png)

# Team Mindrake
*"For death or glory"*


This repository contains agents for winning the CybORG-CAGE [Challenge(s)](https://github\.com/cage\-challenge/cage\-challenge\-1/tree/main/CybORG).

Our blue agent is developed using a hierarchical reinforcement learning approach which comprises two specialised defensive blue agents and a selection agent that decides the best 'agent for the job' based on a given observation. Our two specialists are trained to defend against the red meander and red b\_line agents using the Proximal Policy Optimization (PPO) algorithm.

An important factor in the performance of our specialised agents is that we have allowed them to learn an intrinsic internal reward using curiosity. Curiosity allows our agents to build an internal reward signal which improves performance where rewards from the environment are sparse. Our testing revealed substantial improvements over vanilla PPO where curiosity was applied and also improved upon our attempts to create a more dense reward function manually.

The second important contribution of our solution is a hierarchical approach in which we trained, also using PPO but without curiosity, a selection agent to choose between our specialised agents. We experimented with several different approaches but found best performance using an observation space of one step of the underlying CybORG environment. 


## Current Models:

- Advantage Actor-Critic (A2C, A2C+Curiosity)
- Proximal Policy Optimization (PPO, PPO+Curiosity)
- Importance Weighted Actor-Learner Architecture (IMPALA)
- Deep Q Networks (DQN, Rainbow)
- Distributed Prioritized Experience Replay (Ape-X DQN)
- **Not working yet -** Soft Actor Critic (SAC)

## Setup and Installation

### Install [CAGE Challenge](https://github.com/cage-challenge/cage-challenge-1)

```
# Grab the repo
git clone https://github.com/cage-challenge/cage-challenge-1.git

# from the cage-challenge-1/CybORG directory
pip install -e .
```

### Install our requirements

```
pip install -r requirements.txt
```




## What's in this Repo?

`agents/`: Directory for all agent code

`notebooks/`: Directory for Jupyter notebooks used during prototyping of visualisations and log processing

`config.py`: configure for all commandline flags available

`guild.yml`: Guild.AI configuration file to track results and hyperparameters used during runs


`results/`:

- Directory for results from executing training or evaluation scripts
- Results are summarised in the [results table](results.MD).
- **Instructions for submitting responses to CAGE:**
	- Successful blue agents and any queries re the challenge can be submitted via email to: cage\.aco\.challenge@gmail\.com
	- When submitting a blue agent, teams should include:
		- A team name and contact details\.
		- The code implementing the agent, with a list of dependencies\.
		- A description of your approach in developing a blue agent\.
		- The files and terminal output of the evaluation function\.

For the remainder of the files, see below.



## How to run this code

We have [Ray's RLLib](https://docs.ray.io/en/latest/rllib-algorithms.html) to identify

```
# List of agents to select from is maintained in agents_list.py
python evaluation.py --name-of-agent <agent>
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



