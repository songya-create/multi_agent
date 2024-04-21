
import random
import torch

from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
 writer_reward = SummaryWriter("draw/add_scalar")
 writer_reward.add_scalar(tag = 'average reward/episode',scalar_value ='average reward',global_step = 'episode')