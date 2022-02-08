#!/bin/bash

log_dir="$(pwd)/logs/"
agents_dir="$(pwd)/agents/"


case $1 in

  a2c)
	cd $agents_dir
    python agents/ppo/train_a2c.py $log_dir
    ;;

  ppo)
	cd $agents_dir
    python agents/ppo/train_ppo.py $log_dir
    ;;

  dqn)
	cd $agents_dir
    python agents/ppo/train_dqn.py $log_dir
    ;;

  *)
    echo "ERROR: Invalid model name provided :P"
    ;;
esac





