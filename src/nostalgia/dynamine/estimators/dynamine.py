import os
import math
import torch
import numpy as np

from torch import nn
from nostalgia.dynamine.data.generate import MultivariateTimeSeriesDataloader
from nostalgia.dynamine.configs.dynamine_configs import DynaMineConfig
from nostalgia.dynamine.estimators.ema import ema_loss

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DynaMine(nn.Module):

    def __init__(self, T, config:DynaMineConfig, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = config.loss
        self.alpha = config.alpha
        self.method = 'concat'
        self.iters = config.iters
        self.lr = config.learning_rate
        self.max_number_of_time_steps = config.data.max_number_of_time_steps
        self.T = T

    def forward(self, x, z, z_marg=None,time=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z,time).mean()
        t_marg = self.T(x, z_marg,time)

        if self.loss in ['mine_orig']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None,time=None):
        with torch.no_grad():
            mi = -self.forward(x, z, z_marg,time)
        return mi

    def optimize(self, dataloader:MultivariateTimeSeriesDataloader, opt=None):
        mi_timeseries = []
        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        for iter in range(1, self.iters + 1):
            print(f"Iiter Iindex {iter}")
            mu_mi = 0
            for timeseries_batch in dataloader.train():

                timeseries_batch = timeseries_batch[0]
                batch_size = timeseries_batch.size(0)
                random_time_indexes = torch.randint(0, self.max_number_of_time_steps, (batch_size,))
                x = timeseries_batch[:, 0, :]
                y = timeseries_batch[range(batch_size), random_time_indexes, :]

                opt.zero_grad()
                loss = self.forward(x, y, time=random_time_indexes)
                loss.backward()
                opt.step()

                mu_mi -= loss.item()

        #=========================================
        # FULL TIME SERIES
        #=========================================
        if hasattr(dataloader,"exact_mi"):
            for time_steps_ahead in range(self.max_number_of_time_steps):
                X = dataloader.timeseries[:,0,:]
                Y = dataloader.timeseries[:,time_steps_ahead,:]
                time = torch.full((X.size(0),),time_steps_ahead)
                final_mi = self.mi(X,Y,time=time)
                mi_timeseries.append(final_mi)
                print(f"Final MI: {final_mi}")
                print(f"Exact MI: {dataloader.exact_mi[time_steps_ahead]}")
        return final_mi

