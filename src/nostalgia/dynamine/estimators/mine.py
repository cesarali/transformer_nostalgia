import os
import math
import torch
import numpy as np

from torch import nn
from dataclasses import dataclass
from nostalgia.dynamine.data.generate import MultivariateTimeSeriesDataloader
from nostalgia.dynamine.configs.mine_config import MineConfig
from nostalgia.dynamine.estimators.ema import ema_loss


class Mine(nn.Module):

    def __init__(self, T, config:MineConfig, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = config.loss
        self.alpha = config.alpha
        self.method = 'concat'
        self.iters = config.iters
        self.lr = config.learning_rate

        self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine_orig']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, dataloader:MultivariateTimeSeriesDataloader, opt=None):
        mi_timeseries = []
        for time_steps_ahead in range(1, config.data.max_number_of_time_steps):
            print(time_steps_ahead)
            if opt is None:
                opt = torch.optim.Adam(self.parameters(), lr=self.lr)

            for iter in range(1, self.iters + 1):
                mu_mi = 0
                for timeseries_batch in dataloader.train():
                    timeseries = timeseries_batch[0]
                    x, y = timeseries[:, 0, :], timeseries[:, time_steps_ahead, :]
                    opt.zero_grad()
                    loss = self.forward(x, y)
                    loss.backward()
                    opt.step()

                    mu_mi -= loss.item()
            #=========================================
            # FULL TIME SERIES
            #=========================================
            X = dataloader.timeseries[:,0,:]
            Y = dataloader.timeseries[:,time_steps_ahead,:]
            final_mi = self.mi(X, Y)
            mi_timeseries.append(final_mi)
            print(f"Final MI: {final_mi}")
            print(f"Exact MI: {dataloader.exact_mi[time_steps_ahead]}")
        return final_mi


if __name__=="__main__":
    from nostalgia.dynamine.data.generate import MultivariateTimeSeriesDataloader
    from nostalgia.dynamine.estimators.statistic_networks import StatisticsNetwork

    config = MineConfig()
    config.data.sample_size = 2000
    dataloader = MultivariateTimeSeriesDataloader(config.data)

    statistics_network = StatisticsNetwork(config)
    mine = Mine(T=statistics_network,config=config)
    mi = mine.optimize(dataloader)
