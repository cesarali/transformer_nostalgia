import os
import math
import torch
import numpy as np
import csv
import datetime

from torch import nn
from nostalgia.dynamine.data.generate import MultivariateTimeSeriesDataloader
from nostalgia.dynamine.configs.dynamine_configs import DynaMineConfig
from nostalgia.dynamine.estimators.ema import ema_loss
from .utils import get_device

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DynaMine(nn.Module):

    def __init__(self, T, config:DynaMineConfig, method=None):
        super().__init__()
        self.device = get_device()
        self.to(self.device)
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

        # Define the CSV filename
        csv_filename = "scripts/evaluations/performance_dynamine/experiment_log.csv"

        print(f'Sample size: {dataloader.config.sample_size}')
        print(f'Dimension: {dataloader.config.dimension}')

       # Open the CSV file once and write headers if needed
        file_exists = os.path.isfile(csv_filename)
        csvfile = open(csv_filename, 'a', newline='')
        writer = csv.DictWriter(csvfile, fieldnames=[
            'timestamp', 'sample_size', 'training_proportion', 'batch_size', 'dt', 'jitter',
            'number_of_timesteps', 'max_number_of_time_steps', 'p', 'l', 'sigma_0', 'dimension',
            'time_steps_ahead', 'exact_mi', 'iter', 'estimated_mi'])
        if not file_exists:
            writer.writeheader()

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        for iter in range(1, self.iters + 1):
            if iter%100==0:
                print(f"Iter Index {iter}")
            mu_mi = 0
            for timeseries_batch in dataloader.train():
                timeseries_batch = timeseries_batch[0].to(self.device)
                batch_size = timeseries_batch.size(0)
                random_time_indexes = torch.randint(0, self.max_number_of_time_steps, (batch_size,)).to(self.device)
                x = timeseries_batch[:, 0, :].to(self.device)
                y = timeseries_batch[range(batch_size), random_time_indexes, :].to(self.device)

                opt.zero_grad()
                loss = self.forward(x, y, time=random_time_indexes)
                loss.backward()
                opt.step()

                mu_mi -= loss.item()

            #=========================================
            # FULL TIME SERIES
            #=========================================
            if hasattr(dataloader, "exact_mi"):
                for time_steps_ahead in range(self.max_number_of_time_steps):
                    X = dataloader.timeseries[:, 0, :].to(self.device)
                    Y = dataloader.timeseries[:, time_steps_ahead, :].to(self.device)
                    time = torch.full((X.size(0),), time_steps_ahead).to(self.device)
                    final_mi = self.mi(X, Y, time=time)
                    mi_timeseries.append(final_mi)

                    # Prepare the data to be logged
                    log_data = {
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'sample_size': dataloader.config.sample_size,
                        'training_proportion': dataloader.config.training_proportion,
                        'batch_size': dataloader.config.batch_size,
                        'dt': dataloader.config.dt,
                        'jitter': dataloader.config.jitter,
                        'number_of_timesteps': dataloader.config.number_of_timesteps,
                        'max_number_of_time_steps': dataloader.config.max_number_of_time_steps,
                        'p': str(dataloader.config.p),
                        'l': str(dataloader.config.l),
                        'sigma_0': str(dataloader.config.sigma_0),
                        'dimension': dataloader.config.dimension,
                        'time_steps_ahead': time_steps_ahead,
                        'exact_mi': str(dataloader.exact_mi[time_steps_ahead].item()),
                        'iter': iter,
                        'estimated_mi': final_mi.item()
                    }

                    # print(log_data)

                                # Write the log data to the CSV file
                    with open(csv_filename, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
                        writer.writerow(log_data)

        return mi_timeseries
