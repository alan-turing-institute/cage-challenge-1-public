# from agents.helloworld_agent import TorchCustomModel as BasicAgent # Example
from agents.a2c.a2c.a2c_agent import A2CAgent
from train_rllib_alt import CybORGAgent
# from agents.ppo.PPO import PPO as PPOAgent
# from agents.dqn.dqn_agent import Agent as DQNAgent

AGENTS = {#'rnd': BasicAgent, # Example, maps to BlueLoadAgent
            'a2c-rnd':A2CAgent, 
            'a2c':A2CAgent,
            'rllib-alt':CybORGAgent} 
            #'ppo':PPOAgent} #,
            # 'vas-ppo'=ppoAgent}


from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

RED_AGENTS = {
               'B_lineAgent': B_lineAgent,
               'RedMeanderAgent': RedMeanderAgent,
               'SleepAgent': SleepAgent,
}