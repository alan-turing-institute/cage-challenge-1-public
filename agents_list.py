from agents.helloworld_agent import TorchCustomModel as BasicAgent # Example
from agents.a2c.a2c.a2c_agent import Agent as A2CAgent
from agents.ppo.PPO import PPO as PPOAgent


AGENTS = dict('rnd'=BasicAgent, # Example, maps to BlueLoadAgent
                'mf-a2c-rnd'=A2CAgent, 
                'mf-a2c'=A2CAgent, 
                'mf-ppo'=PPOAgent) #,
                # 'vas-ppo'=ppoAgent)