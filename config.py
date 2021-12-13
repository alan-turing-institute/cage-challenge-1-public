import argparse

import numpy as np
import torch

from agents_list import AGENTS

def configure():
    parser = argparse.ArgumentParser(description='CAGE Challenge Agent config')

    # General flags
    parser.add_argument('--name', type=str, default='ATI', help='Name of person running this')
    parser.add_argument('--team', type=str, default='ATI',  help='Team name...')
    parser.add_argument('--name-of-agent', type=str, default='rnd', choices=list(AGENTS.keys()), help='Name of the agent')
    parser.add_argument('--scenario', type=str, default='Scenario1b', choices=['Scenario1b'], metavar='s', help='Selected Scenario')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')

    # Training/Testing flags
    parser.add_argument('--test', action='store_true', help='Test benchmarks')
    parser.add_argument('--train', action='store_true', help='Train benchmarks')
    parser.add_argument('--batch-size', type=int, default=100, metavar='B', help='Batch size')
    parser.add_argument('--update-step', type=int, default=50, help='go see MF A2C agent docs...')
    parser.add_argument('--episode-length', type=int, default=100, help='go see MF A2C agent docs...')
    parser.add_argument('--training-length', type=int, default=4000, help='go see MF A2C agent docs...')

    # Agent flags
    # # MF A2C Agent
    parser.add_argument('--gamma', type=float, default=0.9, help='go see MF A2C agent docs...')
    parser.add_argument('--epsilon', type=float, default=1.0, help='go see MF A2C agent docs...')
    parser.add_argument('--learning-rate', type=float, default=0.005, help='go see MF A2C agent docs...')
    parser.add_argument('--priority', action='store_true', help='go see MF A2C agent docs...')
    parser.add_argument('--exploring-steps', type=int, default=100, help='go see MF A2C agent docs')
    parser.add_argument('--rnd', action='store_true', help='go see MF A2C agent docs...')
    parser.add_argument('--pre-obs-norm', type=int, default=10, help='go see MF A2C agent docs...')

    args = parser.parse_args()
    return args