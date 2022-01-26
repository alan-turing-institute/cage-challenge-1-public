
![mindrake](https://user-images.githubusercontent.com/10000317/150498045-b712992c-b569-4654-a35e-65660df3f795.png)

# Team Mindrake
*"For death or glory"*


This repository contains agents for winning the CybORG-CAGE [Challenge(s)](https://github\.com/cage\-challenge/cage\-challenge\-1/tree/main/CybORG).

**Important Dates:**

- 20 August 2021: Challenge announced at ACD 2021 and open for submissions\.
- September/October 2021: Release of an additional version of the Challenge problem with a Misinform action available to the blue agent\.
- 1 February 2022: Challenge closed\.
- 4 February 2022: Final results announced\.

## Setup and Installation

```
pip install -r requirements.txt
```

### Install [CAGE Challenge](https://github.com/cage-challenge/cage-challenge-1)

```
# Grab the repo
git clone https://github.com/cage-challenge/cage-challenge-1.git

# from the cage-challenge-1/CybORG directory
pip install -e .
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

We have two methods to train and evaluate agents: [Guild.AI](https://guild.ai/) and Python3. Due to conflict between Guild and [Ray's RLLib](https://docs.ray.io/en/latest/rllib-algorithms.html), these agents must be executed through separate scripts.

### Guild.AI Tracking

```
# List of agents to select from is maintained in agents_list.py
python evaluation.py --name-of-agent <agent>
```

This repository is setup to run trials via [Guild.AI](https://my.guild.ai/)

```
# Run the training script with all defaults for model `a2c`
guild run a2c:train

# Run the evaluation script with all defaults and log into Guild.AI's template
guild run evaluation.py

# View results on TensorBoard (runs local webpage)
guild view

# See what options there are to run trials
guild operations
```





