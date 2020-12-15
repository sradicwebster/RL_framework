import matplotlib.pyplot as plt
import numpy as np
import torch


# return per episode plot as moving average
def plot_moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    reward_move_ave = ret[n - 1:] / n

    plt.plot(reward_move_ave)
    plt.xlabel("Episodes")
    plt.ylabel("Reward per episode")
    plt.xlim(0, len(reward_move_ave))
    plt.ylim(0, reward_move_ave.max())


def discounted_cumsum(input, discount):
    discounted_input = input * discount
    return torch.stack([sum(discounted_input[i:]) for i in range(len(discounted_input))])
