import argparse

import numpy as np
import torch

from agents_list import AGENTS

def configure():
    parser = argparse.ArgumentParser(description='CAGE Challenge Agent config')

    # General flags
    parser.add_argument('--name', type=str, default='ATI', help='Name of person running this')
    parser.add_argument('--team', type=str, default='ATI',  help='Team name...')
    parser.add_argument('--name-of-agent', type=str, default='a2c', choices=list(AGENTS.keys()), help='Name of the agent')
    parser.add_argument('--scenario', type=str, default='Scenario1b', choices=['Scenario1', 'Scenario1b'], metavar='s', help='Selected Scenario')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--has_continuous_action_space', action='store_false', help='continuous action space; else discrete')
  
    # Training/Testing flags
    parser.add_argument('--test', action='store_true', help='Test benchmarks')
    parser.add_argument('--train', action='store_true', help='Train benchmarks')
    parser.add_argument('--batch-size', type=int, default=100, metavar='B', help='Batch size')
    parser.add_argument('--update-step', type=int, default=50, help='See MF A2C agent docs...')
    parser.add_argument('--episode-length', type=int, default=100, help='See MF A2C agent docs...')
    parser.add_argument('--training-length', type=int, default=4000, help='See MF A2C agent docs...')

    # Agent flags
    parser.add_argument('--gamma', type=float, default=0.9, help='See MF A2C agent docs...')
    # # A2C Agent
    parser.add_argument('--epsilon', type=float, default=0.001, help='See MF A2C agent docs...')
    parser.add_argument('--learning-rate', type=float, default=7e-4, help='See MF A2C agent docs...')
    parser.add_argument('--priority', action='store_true', help='See MF A2C agent docs...')
    parser.add_argument('--exploring-steps', type=int, default=100, help='See MF A2C agent docs')
    parser.add_argument('--rnd', action='store_true',help='See MF A2C agent docs...')
    parser.add_argument('--attention', action='store_true', help='See MF A2C agent docs...')
    parser.add_argument('--pre-obs-norm', type=int, default=10, help='See MF A2C agent docs...')
    parser.add_argument('--action-space', type=int, default=54, help='See MF A2C agent docs...')
    parser.add_argument('--obs-space', type=int, default=52,help='See MF A2C agent docs...')
    parser.add_argument('--val-loss-coef', type=float, default=0.5,help='See MF A2C agent docs...')
    parser.add_argument('--entropy-coef', type=float, default=0.0,help='See MF A2C agent docs...')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,help='See MF A2C agent docs...')
    parser.add_argument('--alpha', type=float, default=0.99,help='See MF A2C agent docs...')
    parser.add_argument('--update_prop', type=float, default=0.25,help='See MF A2C agent docs...')
    parser.add_argument('--processes', type=int, default=4,help='See MF A2C agent docs...')

    # # PPO Agent
    parser.add_argument('--action-std', type=float, default=0.6, help='starting std for action distribution (Multivariate Normal)')
    parser.add_argument('--action-std-decay-rate', type=float, default=0.05, help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
    parser.add_argument('--min-action-std', type=float, default=0.1, help='minimum action_std (stop decay after action_std <= min_action_std)')    
    parser.add_argument('--action-std-decay-freq', type=int, default=int(2.5e5), help='action_std decay frequency (in num timesteps)')
    parser.add_argument('--K_epochs', type=int, default=80, help='update policy for K epochs in one PPO update')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    parser.add_argument('--lr_actor', type=float, default=0.0003, help='learning rate for actor network')
    parser.add_argument('--lr_critic', type=float, default=0.001, help='learning rate for critic network')

    args = parser.parse_args()
    print("args", args)
    return args