# from agents.helloworld_agent import TorchCustomModel as BasicAgent # Example
from agents.a2c.a2c.a2c_agent import A2CAgent
# from agents.ppo.PPO import PPO as PPOAgent
# from agents.dqn.dqn_agent import Agent as DQNAgent


AGENTS = {#'rnd': BasicAgent, # Example, maps to BlueLoadAgent
            'a2c-rnd':A2CAgent, 
            'a2c':A2CAgent}#, 
            #'ppo':PPOAgent} #,
            # 'vas-ppo'=ppoAgent}