from packaging import version
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import tensorboard as tb
from tensorflow.python.summary.summary_iterator import summary_iterator

hier_average_values = []
hier_timesteps = []
for summary in summary_iterator("log_dir/controller_1_step/PPO_HierEnv_ebf84_00000_0_2022-02-01_10-30-05/events.out.tfevents.1643711405.Myless-MBP"):
    #print(summary)
    if len(summary.summary.value) > 0 and summary.summary.value[0].tag == 'ray/tune/episode_reward_mean':
        value = summary.summary.value[0].simple_value
        hier_timesteps.append(summary.step/10**6)
        hier_average_values.append(value)




# b_line eval results vs random red agent (500 episodes)
#Average reward vs random (P=0.5) red agent and steps 100 is: -39.009800000000006 with a standard deviation of 36.35361956216385
bline_average_values = [-39.00 for _ in hier_timesteps]

# meander eval reuslts vs random red agent (500 episodes)
# Average reward vs random (P=0.5) red agent and steps 100 is: -23.334400000000002 with a standard deviation of 33.51796061827175
meander_average_values = [-23.34 for _ in hier_timesteps]

# Blue Chosen randomly vs random red agent (500 episodes)
#Average reward vs random (P=0.5) red agent and steps 100 is: -37.870400000000004 with a standard deviation of 25.540283789184176bline_average_values = [-38.84 for _ in hier_timesteps]
random_average_values = [-37.87 for _ in hier_timesteps]

# Blue chosen by our final, submitted hierarchical controller vs random red agent (500 episodes)
# Average reward vs random (P=0.5) red agent and steps 100 is: -8.602599999999999 with a standard deviation of 6.043822368140486
submission_average_values = [-8.60 for _ in hier_timesteps]

# Forr the ‘’ideal controller’' which always chooses correctly:
# Average reward vs random (P=0.5) red agent and steps 100 is: -8.1622 with a standard deviation of 6.205343159524292
ideal_average_values = [-8.16 for _ in hier_timesteps]


plt.plot(hier_timesteps, ideal_average_values, 'b:')
plt.plot(hier_timesteps, submission_average_values, 'b--')
plt.plot(hier_timesteps, hier_average_values)
plt.plot(hier_timesteps, meander_average_values, '--')
plt.plot(hier_timesteps, random_average_values, '--')
plt.plot(hier_timesteps, bline_average_values, '--')

plt.title('Hierarchical controller performance in CybORG')
plt.xlabel('Time step (1e10)')
plt.ylabel('Reward')
plt.grid()
plt.legend(['"Ideal" Controller', 'Submitted Controller', 'PPO Controller Training', 'Meander Defender Only', 'Random Defender', 'Bline Defender Only'])
plt.show()
